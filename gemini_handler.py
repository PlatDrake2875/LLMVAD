import os

from google import genai  # type: ignore[attr-defined]
from google.genai import Client


def get_client() -> Client:
    """Initialize and return a Google Gemini client."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_credentials.json"

    client = genai.Client(
        vertexai=True,
        project="devtest-autopilot",
        location="us-central1",
    )

    return client
