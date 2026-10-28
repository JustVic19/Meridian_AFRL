"""Language-model boundary. Tests use the mock and never send a network request."""

from dataclasses import dataclass
from typing import Protocol


class ModelProviderError(Exception):
    """The language model did not return a completion."""


@dataclass(frozen=True)
class ModelRequest:
    system_prompt: str
    user_prompt: str
    prompt_version: str


@dataclass(frozen=True)
class ModelResponse:
    content: str
    model_name: str
    model_version: str


class LanguageModel(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the model's raw text for this prompt."""


class MockLanguageModel:
    """Return a scripted completion, or raise the supplied provider error."""

    def __init__(
        self,
        content: str = "",
        *,
        error: Exception | None = None,
        model_name: str = "mock",
        model_version: str = "test-1",
    ) -> None:
        self.requests: list[ModelRequest] = []
        self._content = content
        self._error = error
        self._model_name = model_name
        self._model_version = model_version

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return ModelResponse(
            content=self._content,
            model_name=self._model_name,
            model_version=self._model_version,
        )
