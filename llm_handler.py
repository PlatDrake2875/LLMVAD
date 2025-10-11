# filepath: c:\Users\LDG\LLMVAD\gemini_handler.py
import os
from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol

import torch
from dotenv import load_dotenv
from google.genai import Client, types
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

load_dotenv()


class LLMResponse(Protocol):
    """Protocol for LLM response objects."""

    @property
    def text(self) -> str | None:
        """Get the text content of the response."""
        ...


class LLMHandler(ABC):
    """Abstract base class for LLM handlers."""

    model_name: str

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
    """Handler for HuggingFace models (Gemma-3 friendly)."""

    def __init__(self, model_name: str, api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        print(f"Loading model {self.model_name} on {self.device}...")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=self.api_key
        )

        # Model (prefer bfloat16 on modern GPUs; else float16; else float32)
        use_bf16 = (self.device == "cuda") and torch.cuda.is_bf16_supported()
        dtype = (
            torch.bfloat16
            if use_bf16
            else (torch.float16 if self.device == "cuda" else torch.float32)
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            token=self.api_key,
        ).eval()

        # Ensure pad_token_id is set (some chat models omit it)
        if self.tokenizer.pad_token_id is None:
            # fall back to eos as pad, a common practice
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Make greedy the default (you can override in generate_content)
        try:
            self.model.generation_config.do_sample = False
        except AttributeError:
            # Some models don't have generation_config, which is fine
            pass

        print(f"Model loaded successfully on {self.device}")

    def _eos_ids(self):
        """
        Build a robust eos list that includes Gemma's turn-end token.
        """
        ids = set()
        if self.tokenizer.eos_token_id is not None:
            # may be int or list already
            if isinstance(self.tokenizer.eos_token_id, int):
                ids.add(self.tokenizer.eos_token_id)
            else:
                ids.update(self.tokenizer.eos_token_id)

        # Add common chat end markers if they exist in the vocab
        for tok in ("<end_of_turn>", "<|eot_id|>", "<eos>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)

        return list(ids) if ids else None

    def generate_content(
        self,
        video_data: bytes,  # ignored for text-only models like gemma3_text
        system_prompt: str,
        user_prompt: str,
        response_mime_type: str = "application/json",
        *,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Generate using the tokenizer's chat template (Gemma-3 friendly).
        """
        # Gemma-3 expects messages shaped like a batch of conversations.
        # We pass a single conversation (list) wrapped in a batch list.
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ]
        ]

        # Let the tokenizer format + tokenize per the model card
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        eos_ids = self._eos_ids()

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        # Deterministic by default; enable sampling explicitly if wanted
        if do_sample:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Batch decode; strip the prompt portion
        # When using apply_chat_template(tokenize=True, return_dict=True),
        # outputs already include the prompt; cutting the new tokens is optional.
        # tokenizer.batch_decode handles it well; we still remove special tokens.
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # First (and only) item from the batch
        text = decoded[0].strip()

        # Optional safety: stop at well-known markers (should already be handled by eos)
        for marker in ("<end_of_turn>", "<|eot_id|>"):
            if marker in text:
                text = text.split(marker)[0].strip()
                break

        class HuggingFaceResponse:
            def __init__(self, text: str):
                self._text = text

            @property
            def text(self) -> str | None:
                return self._text

        return HuggingFaceResponse(text)

    def get_client(self):
        return self.model


class VLLMHandler(LLMHandler):
    """Handler for vLLM inference engine."""

    def __init__(
        self, model_name: str, api_key: str | None = None, **kwargs: Any
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.llm = None
        self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs: Any) -> None:
        """Initialize the vLLM engine."""
        print(f"Loading model {self.model_name} with vLLM...")

        if self.api_key:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.api_key

        vllm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": 1,
            "enforce_eager": True,
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.7),
            "max_model_len": kwargs.get("max_model_len", 2048),
            "trust_remote_code": True,
            "disable_log_stats": True,
        }

        try:
            self.llm = LLM(**vllm_kwargs)
            print(f"Model {self.model_name} loaded successfully with vLLM")
        except Exception as e:
            print(f"Failed to load model with vLLM: {e}")
            raise

    def generate_content(
        self,
        video_data: bytes,
        system_prompt: str,
        user_prompt: str,
        response_mime_type: str = "application/json",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        **kwargs: Any,
    ):
        """Generate content using vLLM."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs,
        )

        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        response_text = outputs[0].outputs[0].text

        class VLLMResponse:
            def __init__(self, text: str):
                self._text = text.strip()

            @property
            def text(self) -> str | None:
                return self._text

        return VLLMResponse(response_text)

    def get_client(self):
        """Get the underlying vLLM engine."""
        return self.llm


class LLMHandlerFactory:
    """Factory class for creating LLM handlers."""

    @staticmethod
    def create_handler(
        provider: Literal["gemini", "huggingface", "vllm"],
        model_name: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> LLMHandler:
        if provider == "gemini":
            return GeminiLLMHandler(model_name=model_name)
        elif provider == "huggingface":
            return HuggingFaceLLMHandler(model_name=model_name, api_key=api_key)
        elif provider == "vllm":
            return VLLMHandler(model_name=model_name, api_key=api_key, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")


def get_client() -> Client:
    """Legacy function for backward compatibility."""
    handler = GeminiLLMHandler()
    return handler.get_client()
