import threading
from typing import Callable, Protocol, TypeVar, cast

import tokenizers  # type: ignore[import-untyped]

T = TypeVar("T")


class EncoderDecoder(Protocol[T]):
    def encoder(self, text: str) -> list[T]: ...

    def decoder(self, tokens: list[T]) -> str: ...


class LazyEncoder:
    def __init__(self) -> None:
        self._tokenizer: tokenizers.Tokenizer | None = None
        self._init_lock = threading.Lock()
        self._init_thread = threading.Thread(target=self._initialize, daemon=True)
        self._init_thread.start()

    def _initialize(self) -> None:
        with self._init_lock:
            if self._tokenizer is None:
                self._tokenizer = tokenizers.Tokenizer.from_pretrained(
                    "Xenova/claude-tokenizer"
                )

    def _ensure_initialized(self) -> None:
        if self._tokenizer is None:
            with self._init_lock:
                if self._tokenizer is None:
                    self._init_thread.join()

    def encoder(self, text: str) -> list[int]:
        self._ensure_initialized()
        assert self._tokenizer is not None, "Couldn't initialize tokenizer"
        return cast(list[int], self._tokenizer.encode(text).ids)

    def decoder(self, tokens: list[int]) -> str:
        self._ensure_initialized()
        assert self._tokenizer is not None, "Couldn't initialize tokenizer"
        return cast(str, self._tokenizer.decode(tokens))


class TokenizerWrapper:
    """Wraps a tokenizers.Tokenizer to implement EncoderDecoder interface.
    
    Provides both EncoderDecoder interface (.encoder/.decoder) and 
    direct tokenizer interface (.encode/.decode) for compatibility.
    """

    def __init__(self, tokenizer: tokenizers.Tokenizer) -> None:
        self._tokenizer = tokenizer

    def encoder(self, text: str) -> list[int]:
        return cast(list[int], self._tokenizer.encode(text).ids)

    def decoder(self, tokens: list[int]) -> str:
        return cast(str, self._tokenizer.decode(tokens))

    # Aliases for code that uses raw tokenizer interface
    def encode(self, text: str) -> list[int]:
        return self.encoder(text)

    def decode(self, tokens: list[int]) -> str:
        return self.decoder(tokens)


def get_default_encoder() -> EncoderDecoder[int]:
    return LazyEncoder()


def wrap_tokenizer(tokenizer: tokenizers.Tokenizer) -> EncoderDecoder[int]:
    """Wrap a raw tokenizers.Tokenizer to implement EncoderDecoder interface."""
    return TokenizerWrapper(tokenizer)
