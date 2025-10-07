import json

from google.genai.types import GenerateContentResponse

from llm_handler import LLMHandler
from prompts import Ontological_Detectives_Prompts, Ontological_Prompts, Simple_Prompts


class AnomalyJudge:
    """Class for judging video anomalies using various LLM approaches."""

    def __init__(self, llm_handler: LLMHandler):
        """Initialize with an LLM handler."""
        self.llm_handler = llm_handler

    def judge_video_simple(
        self,
        video_data: bytes,
        system_prompt: str = Simple_Prompts.SYSTEM_PROMPT_VIDEO_SIMPLE,
        user_prompt: str = "Follow the system prompt and return valid JSON only.",
    ) -> GenerateContentResponse:
        """Simple video judgment."""
        return self.llm_handler.generate_content(
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
    ) -> GenerateContentResponse:
        """Ontological detectives approach for video judgment."""
        detectives = self.llm_handler.generate_content(
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

            detective_scores = self.llm_handler.generate_content(
                video_data=video_data,
                system_prompt=detective_prompt,
                user_prompt=user_prompt,
            )
            if detective_scores.text is None:
                raise ValueError(f"No response text from detective {detective_title}")
            print(detective_scores.text)
            detectives_scores.append(detective_scores)

        combined_prompt = "\n\n".join(
            ds.text for ds in detectives_scores if ds.text is not None
        )

        master_scores = self.llm_handler.generate_content(
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
    ) -> GenerateContentResponse:
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
            subject_score = self.llm_handler.generate_content(
                video_data=video_data,
                system_prompt=subject_prompt,
                user_prompt=user_prompt,
            )
            if subject_score.text is None:
                raise ValueError(f"No response text from subject {subject}")
            print(subject_score.text)
            subject_scores.append(subject_score)

        combined_prompt = "\n\n".join(
            ss.text for ss in subject_scores if ss.text is not None
        )

        master_scores = self.llm_handler.generate_content(
            video_data=video_data,
            system_prompt=system_prompt,
            user_prompt=combined_prompt,
        )

        return master_scores
