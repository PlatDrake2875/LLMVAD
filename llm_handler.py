# filepath: c:\Users\LDG\LLMVAD\gemini_handler.py
import os
from abc import ABC, abstractmethod
from typing import Literal, Protocol

from dotenv import load_dotenv
from google.genai import Client, types

load_dotenv()


class LLMResponse(Protocol):
    """Protocol for LLM response objects."""

    @property
    def text(self) -> str | None:
        """Get the text content of the response."""
        ...


class LLMHandler(ABC):
    """Abstract base class for LLM handlers."""

    @abstractmethod
    def generate_content(
        self,
        video_data: bytes,
        system_prompt: str,
        user_prompt: str,
        response_mime_type: str = "application/json",
    ) -> LLMResponse:
        """Generate content from video data."""
        pass

    @abstractmethod
    def get_client(self) -> object:
        """Get the underlying client."""
        pass


class GeminiLLMHandler(LLMHandler):
    """Handler for Google Gemini LLM."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = self._initialize_client()

    def _initialize_client(self) -> Client:
        """Initialize Gemini client."""
        api_key = os.getenv("GEMINI_API_KEY", "")

        return Client(api_key=api_key)

    def get_client(self) -> Client:
        """Get the Gemini client."""
        return self.client

    def generate_content(
        self,
        video_data: bytes,
        system_prompt: str,
        user_prompt: str,
        response_mime_type: str = "application/json",
    ):
        """Generate content using Gemini model."""
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type=response_mime_type,
            ),
            contents=[
                types.Part.from_bytes(
                    data=video_data,
                    mime_type="video/mp4",
                ),
                user_prompt,
            ],
        )
        return response


class HuggingFaceLLMHandler(LLMHandler):
    """Handler for HuggingFace models - To be implemented."""

    def __init__(self, model_name: str, api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key
        raise NotImplementedError("HuggingFace handler not yet implemented")

    def generate_content(
        self,
        video_data: bytes,
        system_prompt: str,
        user_prompt: str,
        response_mime_type: str = "application/json",
    ):
        raise NotImplementedError("HuggingFace handler not yet implemented")

    def get_client(self):
        raise NotImplementedError("HuggingFace handler not yet implemented")


class LLMHandlerFactory:
    """Factory class for creating LLM handlers."""

    @staticmethod
    def create_handler(
        provider: Literal["gemini", "huggingface"],
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> LLMHandler:
        if provider == "gemini":
            model = model_name or "gemini-2.5-flash"
            return GeminiLLMHandler(model_name=model)
        elif provider == "huggingface":
            if model_name is None:
                raise ValueError("model_name is required for HuggingFace provider")
            return HuggingFaceLLMHandler(model_name=model_name, api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")


def get_client() -> Client:
    """Legacy function for backward compatibility."""
    handler = GeminiLLMHandler()
    return handler.get_client()
