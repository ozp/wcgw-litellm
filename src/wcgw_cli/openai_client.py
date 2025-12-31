import base64
import json
import mimetypes
import os
import subprocess
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import DefaultDict, Optional, cast

import openai
import petname  # type: ignore[import-untyped]
import rich
import tokenizers  # type: ignore[import-untyped]
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel
from typer import Typer

from wcgw.client.bash_state.bash_state import BashState
from wcgw.client.common import CostData, History, Models, OpenAIModels, discard_input

# Cost data for OpenAI official models
OPENAI_COST_DATA: dict[OpenAIModels, CostData] = {
    "gpt-4o-2024-08-06": CostData(
        cost_per_1m_input_tokens=5, cost_per_1m_output_tokens=15
    ),
    "gpt-4o-mini": CostData(
        cost_per_1m_input_tokens=0.15, cost_per_1m_output_tokens=0.6
    ),
}

# Zero cost for non-OpenAI providers (disables cost tracking)
ZERO_COST = CostData(cost_per_1m_input_tokens=0, cost_per_1m_output_tokens=0)
from wcgw.client.memory import load_memory
from wcgw.client.tool_prompts import TOOL_PROMPTS
from wcgw.client.tools import (
    Context,
    ImageData,
    default_enc,
    get_tool_output,
    initialize,
    which_tool,
    which_tool_name,
)

from .openai_utils import get_input_cost, get_output_cost


class Config(BaseModel):
    model: Models
    cost_limit: float
    cost_file: dict[Models, CostData]
    cost_unit: str = "$"


def text_from_editor(console: rich.console.Console) -> str:
    # First consume all the input till now
    discard_input()
    console.print("\n---------------------------------------\n# User message")
    data = input()
    if data:
        return data
    editor = os.environ.get("EDITOR", "vim")
    with tempfile.NamedTemporaryFile(suffix=".tmp") as tf:
        subprocess.run([editor, tf.name], check=True)
        with open(tf.name, "r") as f:
            data = f.read()
            console.print(data)
            return data


def save_history(history: History, session_id: str) -> None:
    myid = str(history[1]["content"]).replace("/", "_").replace(" ", "_").lower()[:60]
    myid += "_" + session_id
    myid = myid + ".json"

    mypath = Path(".wcgw") / myid
    mypath.parent.mkdir(parents=True, exist_ok=True)
    with open(mypath, "w") as f:
        json.dump(history, f, indent=3)


def parse_user_message_special(msg: str) -> ChatCompletionUserMessageParam:
    # Search for lines starting with `%` and treat them as special commands
    parts: list[ChatCompletionContentPartParam] = []
    for line in msg.split("\n"):
        if line.startswith("%"):
            args = line[1:].strip().split(" ")
            command = args[0]
            assert command == "image"
            image_path = " ".join(args[1:])
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                image_type = mimetypes.guess_type(image_path)[0]
                dataurl = f"data:{image_type};base64,{image_b64}"
            parts.append(
                {"type": "image_url", "image_url": {"url": dataurl, "detail": "auto"}}
            )
        else:
            if len(parts) > 0 and parts[-1]["type"] == "text":
                parts[-1]["text"] += "\n" + line
            else:
                parts.append({"type": "text", "text": line})
    return {"role": "user", "content": parts}


app = Typer(pretty_exceptions_show_locals=False)


def _is_openai_official(base_url: Optional[str]) -> bool:
    """Check if base_url points to OpenAI official API."""
    if base_url is None:
        return True
    return base_url.startswith("https://api.openai.com")


def _get_cost_data(model: Models, base_url: Optional[str]) -> CostData:
    """Get cost data for a model. Returns zero cost for non-OpenAI providers."""
    if not _is_openai_official(base_url):
        return ZERO_COST
    # Try to get cost data for known OpenAI models
    return OPENAI_COST_DATA.get(model, ZERO_COST)  # type: ignore[arg-type]


@app.command()
def loop(
    first_message: Optional[str],
    limit: Optional[float],
    resume: Optional[str],
    model: Models,
    base_url: Optional[str],
) -> tuple[str, float]:

    session_id = str(uuid.uuid4())[:6]

    history: History = []
    waiting_for_assistant = False

    memory = None
    if resume:
        try:
            _, memory, _ = load_memory(
                resume,
                24000,  # coding_max_tokens
                8000,   # noncoding_max_tokens
                lambda x: default_enc.encoder(x),
                lambda x: default_enc.decoder(x),
            )
        except OSError:
            if resume == "latest":
                resume_path = sorted(Path(".wcgw").iterdir(), key=os.path.getmtime)[-1]
            else:
                resume_path = Path(resume)
            if not resume_path.exists():
                raise FileNotFoundError(f"File {resume} not found")
            with resume_path.open() as f:
                history = json.load(f)
            if len(history) <= 2:
                raise ValueError("Invalid history file")
            first_message = ""
            waiting_for_assistant = history[-1]["role"] != "assistant"

    my_dir = os.path.dirname(__file__)

    # Get cost data based on model and provider
    cost_data = _get_cost_data(model, base_url)

    config = Config(
        model=model,
        cost_limit=0.1,
        cost_unit="$",
        cost_file={model: cost_data},
    )

    if limit is not None:
        config.cost_limit = limit
    limit = config.cost_limit

    enc = tokenizers.Tokenizer.from_pretrained("Xenova/gpt-4o")

    tools = [
        openai.pydantic_function_tool(
            which_tool_name(tool.name), description=tool.description
        )
        for tool in TOOL_PROMPTS
        if tool.name != "Initialize"
    ]

    cost: float = 0
    input_toks = 0
    output_toks = 0
    system_console = rich.console.Console(style="blue", highlight=False, markup=False)
    error_console = rich.console.Console(style="red", highlight=False, markup=False)
    user_console = rich.console.Console(
        style="bright_black", highlight=False, markup=False
    )
    assistant_console = rich.console.Console(
        style="white bold", highlight=False, markup=False
    )

    with BashState(
        system_console, os.getcwd(), None, None, None, None, True, None
    ) as bash_state:
        context = Context(bash_state, system_console)
        system, context, _ = initialize(
            "first_call",
            context,
            os.getcwd(),
            [],
            resume if (memory and resume) else "",
            24000,  # coding_max_tokens
            8000,   # noncoding_max_tokens
            mode="wcgw",
            thread_id="",
        )

        if not history:
            history = [{"role": "system", "content": system}]
        else:
            if history[-1]["role"] == "tool":
                waiting_for_assistant = True

        # Create client with optional custom base_url for OpenAI-compatible providers
        if base_url is not None:
            client = OpenAI(base_url=base_url)
        else:
            client = OpenAI()

        while True:
            if cost > limit:
                system_console.print(
                    f"\nCost limit exceeded. Current cost: {cost}, input tokens: {input_toks}, output tokens: {output_toks}"
                )
                break

            if not waiting_for_assistant:
                if first_message:
                    msg = first_message
                    first_message = ""
                else:
                    msg = text_from_editor(user_console)

                history.append(parse_user_message_special(msg))
            else:
                waiting_for_assistant = False

            cost_, input_toks_ = get_input_cost(
                config.cost_file[config.model], enc, history
            )
            cost += cost_
            input_toks += input_toks_

            stream = client.chat.completions.create(
                messages=history,
                model=config.model,
                stream=True,
                tools=tools,
            )

            system_console.print(
                "\n---------------------------------------\n# Assistant response",
                style="bold",
            )
            tool_call_args_by_id = DefaultDict[str, DefaultDict[int, str]](
                lambda: DefaultDict(str)
            )
            _histories: History = []
            item: ChatCompletionMessageParam
            full_response: str = ""
            image_histories: History = []
            try:
                for chunk in stream:
                    if chunk.choices[0].finish_reason == "tool_calls":
                        assert tool_call_args_by_id
                        item = {
                            "role": "assistant",
                            "content": full_response,
                            "tool_calls": [
                                {
                                    "id": tool_call_id + str(toolindex),
                                    "type": "function",
                                    "function": {
                                        "arguments": tool_args,
                                        "name": type(which_tool(tool_args)).__name__,
                                    },
                                }
                                for tool_call_id, toolcallargs in tool_call_args_by_id.items()
                                for toolindex, tool_args in toolcallargs.items()
                            ],
                        }
                        cost_, output_toks_ = get_output_cost(
                            config.cost_file[config.model], enc, item
                        )
                        cost += cost_
                        system_console.print(
                            f"\n---------------------------------------\n# Assistant invoked tools: {[which_tool(tool['function']['arguments']) for tool in item['tool_calls']]}"
                        )
                        system_console.print(
                            f"\nTotal cost: {config.cost_unit}{cost:.3f}"
                        )
                        output_toks += output_toks_

                        _histories.append(item)
                        for tool_call_id, toolcallargs in tool_call_args_by_id.items():
                            for toolindex, tool_args in toolcallargs.items():
                                try:
                                    # Create a wrapper for loop_call that captures model/base_url
                                    def loop_wrapper(msg: str, lim: float) -> tuple[str, float]:
                                        return loop(msg, lim, None, model, base_url)

                                    output_or_dones, cost_ = get_tool_output(
                                        context,
                                        json.loads(tool_args),
                                        enc,
                                        limit - cost,
                                        loop_wrapper,
                                        24000,  # coding_max_tokens
                                        8000,   # noncoding_max_tokens
                                    )
                                    output_or_done = output_or_dones[0]
                                except Exception as e:
                                    output_or_done = (
                                        f"GOT EXCEPTION while calling tool. Error: {e}"
                                    )
                                    tb = traceback.format_exc()
                                    error_console.print(output_or_done + "\n" + tb)
                                    cost_ = 0
                                cost += cost_
                                system_console.print(
                                    f"\nTotal cost: {config.cost_unit}{cost:.3f}"
                                )

                                output = output_or_done

                                if isinstance(output, ImageData):
                                    randomId = petname.Generate(2, "-")
                                    if not image_histories:
                                        image_histories.extend(
                                            [
                                                {
                                                    "role": "assistant",
                                                    "content": f"Share images with ids: {randomId}",
                                                },
                                                {
                                                    "role": "user",
                                                    "content": [
                                                        {
                                                            "type": "image_url",
                                                            "image_url": {
                                                                "url": output.dataurl,
                                                                "detail": "auto",
                                                            },
                                                        }
                                                    ],
                                                },
                                            ]
                                        )
                                    else:
                                        image_histories[0]["content"] += ", " + randomId
                                        second_content = image_histories[1]["content"]
                                        assert isinstance(second_content, list)
                                        second_content.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": output.dataurl,
                                                    "detail": "auto",
                                                },
                                            }
                                        )

                                    item = {
                                        "role": "tool",
                                        "content": f"Ask user for image id: {randomId}",
                                        "tool_call_id": tool_call_id + str(toolindex),
                                    }
                                else:
                                    item = {
                                        "role": "tool",
                                        "content": str(output),
                                        "tool_call_id": tool_call_id + str(toolindex),
                                    }
                                cost_, output_toks_ = get_output_cost(
                                    config.cost_file[config.model], enc, item
                                )
                                cost += cost_
                                output_toks += output_toks_

                                _histories.append(item)
                        waiting_for_assistant = True
                        break
                    elif chunk.choices[0].finish_reason:
                        assistant_console.print("")
                        item = {
                            "role": "assistant",
                            "content": full_response,
                        }
                        cost_, output_toks_ = get_output_cost(
                            config.cost_file[config.model], enc, item
                        )
                        cost += cost_
                        output_toks += output_toks_

                        system_console.print(
                            f"\nTotal cost: {config.cost_unit}{cost:.3f}"
                        )
                        _histories.append(item)
                        break

                    if chunk.choices[0].delta.tool_calls:
                        tool_call = chunk.choices[0].delta.tool_calls[0]
                        if tool_call.function and tool_call.function.arguments:
                            tool_call_args_by_id[tool_call.id or ""][
                                tool_call.index
                            ] += tool_call.function.arguments

                    chunk_str = chunk.choices[0].delta.content or ""
                    assistant_console.print(chunk_str, end="")
                    full_response += chunk_str
            except KeyboardInterrupt:
                waiting_for_assistant = False
                input("Interrupted...enter to redo the current turn")
            else:
                history.extend(_histories)
                history.extend(image_histories)
                save_history(history, session_id)

    return "Couldn't finish the task", cost


if __name__ == "__main__":
    app()
