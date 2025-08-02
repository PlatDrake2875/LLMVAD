import os

from google import genai
from google.genai import Client


def get_client():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GeminiAPI/gcp_credentials.json"

    client = genai.Client(
        vertexai=True,
        project="devtest-autopilot",
        location="us-central1",
    )

    return client


def invoke_llm(client: Client, content: str):
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=content
    )
    print(response.text)


if __name__ == "__main__":
    client = get_client()
    invoke_llm(client, "Who are you? But explain it to me like a drunken irish")
