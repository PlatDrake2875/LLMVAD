from eval import EvalModes, invoke_video_understanding_llm
from llm_handler import LLMHandlerFactory

if __name__ == "__main__":
    handler = LLMHandlerFactory.create_handler(
        provider="gemini", model_name="gemini-2.5-flash"
    )

    invoke_video_understanding_llm(
        handler=handler,
        judge_mode=EvalModes.ONTOLOGICAL_CATEGORIES,
        max_concurrent=8,
        max_size=10,
        max_retries=5,
    )
