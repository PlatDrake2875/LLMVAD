# filepath: c:\Users\LDG\LLMVAD\gemini_handler.py
import json
import os
from abc import ABC, abstractmethod
from typing import Literal

from dotenv import load_dotenv
from google.genai import Client, types

from prompts import Ontological_Detectives_Prompts, Ontological_Prompts, Simple_Prompts

# Load environment variables
load_dotenv()


class LLMHandler(ABC):
    """Abstract base class for LLM handlers."""

    @abstractmethod
    def generate_content(
        self,
        video_data: bytes,
        system_prompt: str,
        user_prompt: str,
        response_mime_type: str = "application/json",
    ):
        """Generate content from video data."""
        pass

    @abstractmethod
    def get_client(self):
        """Get the underlying client."""
        pass


class GeminiLLMHandler(LLMHandler):
    """Handler for Google Gemini LLM."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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

    def judge_video_simple(
        self,
        video_data: bytes,
        system_prompt: str = Simple_Prompts.SYSTEM_PROMPT_VIDEO_SIMPLE,
        user_prompt: str = "Follow the system prompt and return valid JSON only.",
    ):
        """Simple video judgment."""
        return self.generate_content(
            video_data=video_data,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_mime_type="application/json",
        )

    def judge_video_ontological_detectives(
        self,
        video_data: bytes,
        system_prompt: str = Ontological_Detectives_Prompts.SYSTEM_PROMPT_ONTOLOGICAL,
        user_prompt: str = "Follow the system prompt.",
    ):
        """Ontological detectives approach for video judgment."""
        detectives = self.generate_content(
            video_data=video_data,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if detectives.text is None:
            raise ValueError("No response text from detectives generation")

        detectives_list = json.loads(detectives.text)
        print(f"Detectives list: {detectives_list}")

        detectives_scores = []
        for detective in detectives_list:
            detective_title = detective.split("<>")[0]
            detective_target = detective.split("<>")[1]
            detective_prompt = (
                Ontological_Detectives_Prompts.get_system_prompt_detective(
                    detective_title, detective_target
                )
            )

            detective_scores = self.generate_content(
                video_data=video_data,
                system_prompt=detective_prompt,
                user_prompt=user_prompt,
            )
            print(detective_scores.text)
            detectives_scores.append(detective_scores)

        combined_prompt = "\n\n".join(ds.text for ds in detectives_scores)

        master_scores = self.generate_content(
            video_data=video_data,
            system_prompt=Ontological_Detectives_Prompts.SYSTEM_PROMPT_MASTER,
            user_prompt=combined_prompt,
        )

        return master_scores

    def judge_video_ontological_categories(
        self,
        video_data: bytes,
        system_prompt: str = Ontological_Prompts.SYSTEM_PROMPT_SYNTHESIZER,
        user_prompt: str = "Follow the system prompt.",
        subjects: list[str] | None = None,
    ):
        """Ontological categories approach for video judgment."""
        if subjects is None:
            subjects = [
                "violence",
                "nature",
                "sports",
                "urban",
                "vehicles",
                "crowds",
                "weapons",
                "emergency",
                "normal activity",
                "religious ritual",
                "culture",
            ]

        subject_scores = []
        for subject in subjects:
            subject_prompt = Ontological_Prompts.get_system_prompt_base(subject)
            subject_score = self.generate_content(
                video_data=video_data,
                system_prompt=subject_prompt,
                user_prompt=user_prompt,
            )
            print(subject_score.text)
            subject_scores.append(subject_score)

        combined_prompt = "\n\n".join(ss.text for ss in subject_scores)

        master_scores = self.generate_content(
            video_data=video_data,
            system_prompt=system_prompt,
            user_prompt=combined_prompt,
        )

        return master_scores


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
