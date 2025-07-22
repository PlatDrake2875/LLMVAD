import logging
from typing import List, Optional

from .gemini_client import GeminiClient


class AnomalyDetector:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        project_id: str = "devtest-autopilot",
    ):
        self.gemini_client = GeminiClient(model_name=model_name, project_id=project_id)
        logging.info(f"GeminiAnomalyDetector initialized with model: {model_name}")

    def judge_video(self, video_descriptions: List[str]) -> Optional[List[float]]:
        if not video_descriptions:
            logging.warning("No video descriptions provided for anomaly detection.")
            return None

        logging.info(
            f"Starting anomaly detection for {len(video_descriptions)} video chunks."
        )

        anomaly_scores = []

        for i, description in enumerate(video_descriptions):
            try:
                score = self._judge_chunk(description, i)
                anomaly_scores.append(score)
                logging.info(
                    f"Chunk {i + 1}/{len(video_descriptions)}: Anomaly score = {score:.4f}"
                )
            except Exception as e:
                logging.error(f"Error judging chunk {i + 1}: {e}")
                anomaly_scores.append(0.0)

        logging.info(
            f"Anomaly detection completed. Average score: {sum(anomaly_scores) / len(anomaly_scores):.4f}"
        )
        return anomaly_scores

    def _judge_chunk(self, description: str, chunk_index: int) -> float:
        prompt = f"""Analyze this video chunk description and rate how anomalous or violent it is on a scale from 0.0 to 1.0, where:
- 0.0 = completely normal, peaceful activity
- 1.0 = extremely violent, dangerous, or anomalous behavior

Video chunk description: {description}

Respond with only a single number between 0.0 and 1.0 representing the anomaly score."""

        try:
            response = self.gemini_client.generate_text(
                prompt, temperature=0.1, max_tokens=50
            )

            score = self._extract_score(response)
            return score

        except Exception as e:
            logging.error(
                f"Error generating anomaly score for chunk {chunk_index}: {e}"
            )
            return 0.0

    def _extract_score(self, response: str) -> float:
        try:
            response = response.strip()
            for line in response.split("\n"):
                line = line.strip()
                if line and any(c.isdigit() for c in line):
                    score_str = "".join(c for c in line if c.isdigit() or c == ".")
                    if score_str:
                        score = float(score_str)
                        return max(0.0, min(1.0, score))
            return 0.0
        except (ValueError, AttributeError):
            logging.warning(f"Could not extract score from response: {response}")
            return 0.0
