import importlib
import os
from typing import Optional

import typer
from dotenv import load_dotenv
from typer import Typer

from wcgw_cli.anthropic_client import loop as claude_loop
from wcgw_cli.openai_client import loop as openai_loop

app = Typer(pretty_exceptions_show_locals=False)

# Default model for OpenAI official
DEFAULT_MODEL = "gpt-4o-2024-08-06"


@app.command()
def loop(
    claude: bool = False,
    first_message: Optional[str] = None,
    limit: Optional[float] = None,
    resume: Optional[str] = None,
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use (e.g., glm-4.7, qwen-235b)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-b", help="OpenAI-compatible API base URL"),
    version: bool = typer.Option(False, "--version", "-v"),
) -> tuple[str, float]:
    if version:
        version_ = importlib.metadata.version("wcgw")
        print(f"wcgw version: {version_}")
        exit()

    # Load .env for environment variable fallbacks
    load_dotenv()

    # Resolve model: Flag > Env > Default
    resolved_model: str = model if model is not None else os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    # Resolve base_url: Flag > Env > None (uses OpenAI official)
    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")

    if claude:
        return claude_loop(
            first_message=first_message,
            limit=limit,
            resume=resume,
        )
    else:
        return openai_loop(
            first_message=first_message,
            limit=limit,
            resume=resume,
            model=resolved_model,
            base_url=resolved_base_url,
        )


if __name__ == "__main__":
    app()
