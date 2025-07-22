import logging
import os
from typing import Any, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(
        self,
        credentials_path: str = "gcp_credentials.json",
        project_id: str = "devtest-autopilot",
        model_name: str = "gemini-2.5-flash",
    ):
        self.credentials_path = credentials_path
        self.project_id = project_id
        self.model_name = model_name

        self._setup_credentials()
        self._init_google_ai_client()

    def _setup_credentials(self):
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found: {self.credentials_path}"
            )

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
            self.credentials_path
        )
        os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id

        logger.info(f"Credentials set up from: {self.credentials_path}")
        logger.info(f"Project ID: {self.project_id}")

    def _init_google_ai_client(self):
        try:
            import google.generativeai as genai

            genai.configure(api_key=None)

            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Google AI client initialized with model: {self.model_name}")

        except ImportError:
            raise ImportError(
                "Google AI Python SDK not installed. "
                "Install with: pip install google-generativeai"
            )

    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 40,
    ) -> str:
        try:
            return self._generate_text_google_ai(
                prompt, max_tokens, temperature, top_p, top_k
            )

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def _generate_text_google_ai(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        response = self.model.generate_content(
            prompt, generation_config=generation_config
        )

        return response.text

    def generate_text_with_safety(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            return self._generate_text_with_safety_google_ai(prompt, **kwargs)

        except Exception as e:
            logger.error(f"Error generating text with safety: {e}")
            raise

    def _generate_text_with_safety_google_ai(
        self, prompt: str, **kwargs
    ) -> Dict[str, Any]:
        response = self.model.generate_content(prompt, **kwargs)

        return {
            "text": response.text,
            "safety_ratings": response.safety_ratings
            if hasattr(response, "safety_ratings")
            else None,
            "prompt_feedback": response.prompt_feedback
            if hasattr(response, "prompt_feedback")
            else None,
        }


if __name__ == "__main__":
    print("=== Gemini Client Example ===")
    try:
        client = GeminiClient()
        response = client.generate_text("Explain quantum computing in simple terms.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
