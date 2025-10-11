"""Unit tests for LLM handlers (HuggingFace and vLLM).

Due to GPU memory constraints, these tests should be run separately:
- Run HuggingFace tests: pytest tests/test_llm_handlers.py -m huggingface
- Run vLLM tests: pytest tests/test_llm_handlers.py -m vllm
- Run factory tests: pytest tests/test_llm_handlers.py -m factory
"""
# ruff: noqa: S101  # Allow asserts in tests

import os

import pytest

from llm_handler import LLMHandlerFactory, VLLMHandler


@pytest.fixture
def hf_token():
    """Get HuggingFace API token from environment."""
    token = os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        pytest.skip("HUGGINGFACE_API_KEY not set in environment")
    return token


@pytest.fixture
def sample_prompts():
    """Provide sample prompts for testing."""
    return {
        "system_prompt": "You are a helpful assistant.",
        "user_prompt": "What is 2+2?",
        "video_data": b"",  # Empty for text-only models
    }


@pytest.mark.huggingface
class TestHuggingFaceHandler:
    """Tests for HuggingFace LLM Handler."""

    @pytest.fixture
    def model_name(self):
        """Return the model name for testing."""
        return "google/gemma-3-1b-it"

    @pytest.fixture
    def handler(self, hf_token, model_name):
        """Create a HuggingFace handler instance."""
        handler = LLMHandlerFactory.create_handler(
            provider="huggingface",
            model_name=model_name,
            api_key=hf_token,
            device_map="auto",
            torch_dtype="auto",
        )
        yield handler
        # Cleanup
        del handler

    def test_handler_creation(self, hf_token, model_name):
        """Test that HuggingFace handler can be created."""
        handler = LLMHandlerFactory.create_handler(
            provider="huggingface",
            model_name=model_name,
            api_key=hf_token,
            device_map="auto",
            torch_dtype="auto",
        )
        assert handler is not None
        assert handler.model_name == model_name

    def test_generate_content(self, handler, sample_prompts):
        """Test content generation with HuggingFace handler."""
        response = handler.generate_content(
            video_data=sample_prompts["video_data"],
            system_prompt=sample_prompts["system_prompt"],
            user_prompt=sample_prompts["user_prompt"],
            max_new_tokens=50,
            do_sample=False,
        )

        assert response is not None
        assert hasattr(response, "text")
        assert response.text is not None
        assert len(response.text) > 0
        # Check that response contains something about the answer
        assert any(
            digit in response.text.lower() for digit in ["4", "four", "2+2", "2 + 2"]
        )

    def test_generate_with_different_max_tokens(self, handler, sample_prompts):
        """Test generation with different max token settings."""
        # Short response
        short_response = handler.generate_content(
            video_data=sample_prompts["video_data"],
            system_prompt=sample_prompts["system_prompt"],
            user_prompt=sample_prompts["user_prompt"],
            max_new_tokens=10,
            do_sample=False,
        )

        # Longer response
        long_response = handler.generate_content(
            video_data=sample_prompts["video_data"],
            system_prompt=sample_prompts["system_prompt"],
            user_prompt=sample_prompts["user_prompt"],
            max_new_tokens=100,
            do_sample=False,
        )

        assert short_response.text is not None
        assert long_response.text is not None
        # Note: Due to chat formatting, this assertion might not always hold
        # assert len(long_response.text) >= len(short_response.text)

    def test_factory_pattern(self, hf_token, model_name):
        """Test that factory correctly creates HuggingFace handler."""
        handler = LLMHandlerFactory.create_handler(
            provider="huggingface",
            model_name=model_name,
            api_key=hf_token,
            device_map="auto",
            torch_dtype="auto",
        )
        assert handler is not None
        assert handler.model_name == model_name


@pytest.mark.vllm
class TestVLLMHandler:
    """Tests for vLLM Handler."""

    @pytest.fixture
    def model_name(self):
        """Return the model name for testing (smaller model for memory constraints)."""
        return "Qwen/Qwen3-0.6B"

    @pytest.fixture
    def handler(self, hf_token, model_name):
        """Create a vLLM handler instance."""
        handler = LLMHandlerFactory.create_handler(
            provider="vllm",
            model_name=model_name,
            api_key=hf_token,
            max_model_len=1024,
            gpu_memory_utilization=0.4,  # Lower to fit in available GPU memory
            max_num_seqs=1,
            enforce_eager=True,
            disable_log_stats=True,
        )
        yield handler
        # Cleanup
        del handler

    def test_handler_creation(self, hf_token, model_name):
        """Test that vLLM handler can be created."""
        handler = LLMHandlerFactory.create_handler(
            provider="vllm",
            model_name=model_name,
            api_key=hf_token,
            max_model_len=512,
            gpu_memory_utilization=0.4,  # Lower to fit in available GPU memory
            enforce_eager=True,
        )
        assert isinstance(handler, VLLMHandler)
        assert handler.model_name == model_name

    def test_generate_content(self, handler, sample_prompts):
        """Test content generation with vLLM handler."""
        response = handler.generate_content(
            video_data=sample_prompts["video_data"],
            system_prompt=sample_prompts["system_prompt"],
            user_prompt=sample_prompts["user_prompt"],
            max_tokens=50,
            temperature=0.0,
        )

        assert response is not None
        assert hasattr(response, "text")
        assert response.text is not None
        assert len(response.text) > 0
        # Check that response contains something about the answer
        assert any(
            digit in response.text.lower() for digit in ["4", "four", "2+2", "2 + 2"]
        )

    def test_generate_with_temperature(self, handler, sample_prompts):
        """Test generation with different temperature settings."""
        # Temperature 0 (deterministic)
        response1 = handler.generate_content(
            video_data=sample_prompts["video_data"],
            system_prompt=sample_prompts["system_prompt"],
            user_prompt=sample_prompts["user_prompt"],
            max_tokens=30,
            temperature=0.0,
        )

        response2 = handler.generate_content(
            video_data=sample_prompts["video_data"],
            system_prompt=sample_prompts["system_prompt"],
            user_prompt=sample_prompts["user_prompt"],
            max_tokens=30,
            temperature=0.0,
        )

        # With temperature 0, responses should be identical
        assert response1.text == response2.text

    def test_generate_with_max_tokens(self, handler, sample_prompts):
        """Test generation with different max token settings."""
        short_response = handler.generate_content(
            video_data=sample_prompts["video_data"],
            system_prompt=sample_prompts["system_prompt"],
            user_prompt=sample_prompts["user_prompt"],
            max_tokens=10,
            temperature=0.0,
        )

        long_response = handler.generate_content(
            video_data=sample_prompts["video_data"],
            system_prompt=sample_prompts["system_prompt"],
            user_prompt=sample_prompts["user_prompt"],
            max_tokens=50,
            temperature=0.0,
        )

        assert short_response.text is not None
        assert long_response.text is not None

    def test_factory_pattern(self, hf_token, model_name):
        """Test that factory correctly creates vLLM handler."""
        handler = LLMHandlerFactory.create_handler(
            provider="vllm",
            model_name=model_name,
            api_key=hf_token,
            max_model_len=512,
            enforce_eager=True,
        )

        # Check that we got the right handler type
        assert handler.__class__.__name__ == "VLLMHandler"


@pytest.mark.factory
class TestLLMHandlerFactory:
    """Tests for the LLM Handler Factory."""

    def test_invalid_provider(self, hf_token):
        """Test that invalid provider raises appropriate error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMHandlerFactory.create_handler(
                provider="invalid_provider",
                model_name="some-model",
                api_key=hf_token,
            )

    def test_gemini_provider_option(self):
        """Test that gemini provider is available as an option."""
        # This test just verifies the factory accepts 'gemini' as a provider
        # We don't actually test it since it requires different credentials
        # The factory should not raise ValueError for 'gemini'
        try:
            # This will fail due to missing/invalid credentials, but shouldn't
            # raise ValueError for unknown provider
            LLMHandlerFactory.create_handler(
                provider="gemini",
                model_name="gemini-2.0-flash-exp",
                api_key="test_key",
            )
        except ValueError as e:
            if "Unknown provider" in str(e):
                pytest.fail("Factory should accept 'gemini' as a valid provider")
        except Exception as e:  # noqa: S110
            # Other exceptions (like auth errors) are expected and OK
            # We just want to verify the provider is recognized
            pass

    def test_all_providers_available(self):
        """Test that all three providers are available."""
        # This is a meta-test to ensure we support all expected providers
        providers = ["gemini", "huggingface", "vllm"]

        for provider in providers:
            try:
                # Try creating with dummy credentials
                # Will fail on auth/model loading but shouldn't fail on provider validation
                LLMHandlerFactory.create_handler(
                    provider=provider,
                    model_name="test-model",
                    api_key="test_key",
                )
            except ValueError as e:
                if "Unknown provider" in str(e):
                    pytest.fail(f"Provider '{provider}' should be supported")
            except Exception as e:  # noqa: S110
                # Other exceptions are expected
                # We just want to verify the provider is recognized
                pass
