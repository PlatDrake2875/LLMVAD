import logging
import os
import re

import torch
import torch._dynamo
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from .hf_auth import get_hf_token

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
torch.backends.cudnn.enabled = False
torch.set_float32_matmul_precision("high")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


class AnomalyDetector:
    """
    A class to detect anomalies in video summaries using HuggingFace's Gemma 3 model.

    This class uses a local Gemma 3 model to judge given video summaries
    for anomalous events and extracts a numerical anomaly score.
    """

    def __init__(self, model_name: str = "google/gemma-3-4b-it", device: str = "auto"):
        """
        Initializes the AnomalyDetector with model configuration and prompts.
        """
        self.model_name = model_name
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        hf_token = get_hf_token()

        self.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            attn_implementation="eager",
            token=hf_token,
        ).eval()

        self.messages = []

        self.prefix_prompt = "Analyze this video scene description "
        self.judge_prompts = {
            "Judge": "for any unusual or anomalous behavior.",
            "Scene": "checking for anything suspicious in the background.",
            "Action": "for anomalous actions, activities or behaviour.",
        }

        self.master_prompt = {
            "The following descriptions represent different view-points from which different people evaluated the same setting in terms of anomalous behaviour."
            "Your task is to judge those reasonings and decide which one describes the truth better in the given context."
            "A score of 0.0 means normal behaviour, "
            "0.1-0.3 means barely suspicious activity, "
            "0.3-0.5 means somewhat anomalous behaviour,"
            "0.5-0.7 means likely anomalous behaviour, "
            ">0.7 means highly anomalous behaviour."
            "Provide your score as a decimal number (e.g., 0.3 or 0.8). "
            "Score: "
        }

        self.system_prompt = (
            "Rate the anomaly level in this video scene description on a scale from 0.0 to 1.0, "
            "where 0.0 means normal behaviour, "
            "0.1-0.3 means barely suspicious activity, "
            "0.3-0.5 means somewhat anomalous behaviour,"
            "0.5-0.7 means likely anomalous behaviour, "
            ">0.7 means highly anomalous behaviour."
            "Provide your score as a decimal number (e.g., 0.3 or 0.8). "
            "Score: "
        )

        logging.info(
            f"AnomalyDetector initialized with model: {self.model_name}, device: {self.device}"
        )

    def _extract_score_from_response(self, model_response_content: str) -> float:
        """
        Extracts the anomaly score from the LLM's raw text response.
        """
        score_tag_match = re.search(
            r"</?SCORE>([0-9]*\.?[0-9]+)", model_response_content, re.IGNORECASE
        )
        if score_tag_match:
            try:
                score = float(score_tag_match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        score_match = re.search(r"([0-9]*\.?[0-9]+)", model_response_content)
        if score_match:
            try:
                score = float(score_match.group(1))
                if score > 1.0:
                    score = score / 100.0
                return max(0.0, min(1.0, score))
            except ValueError:
                logging.warning(
                    f"Could not convert extracted score to float: '{score_match.group(1)}'"
                )
                return 0.1
        else:
            logging.warning(
                f"No valid float score found in LLM response. Response: '{model_response_content.strip()}'"
            )
            return 0.1

    def _generate_text(
        self, system_prompt: str, prompt: str, max_length: int = 512
    ) -> str:
        """
        Generates text using the Gemma 3 model with conversation context.
        """
        if not self.messages:
            self.messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            ]

        self.messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )

        inputs = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(
            self.model.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        response_content = decoded.strip()

        self.messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response_content}],
            }
        )

        return response_content

    def _master_judge(self):
        query_messages = [
            {"role": "user", "content": [{"type": "text", "text": message}]}
            for message in self.messages[:-3]
        ]
        system_message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.master_prompt}],
            }
        ]

        messages = system_message + query_messages
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(
            self.model.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        response_content = decoded.strip()

        self.messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response_content}],
            }
        )

        return response_content

    def clear_context(self):
        """
        Clears the conversation context.
        """
        self.messages = []
        logging.info("Conversation context cleared.")

    def get_context(self) -> list:
        """
        Returns the current conversation context.
        """
        return self.messages.copy()

    def _display_judgment(self, idx: int, model_response_content: str):
        """
        Logs the LLM's judgment.
        """
        judgment_text = f"\n{'#' * 30}\n ANOMALY JUDGMENT (SUMMARY CHUNK {idx})\n{'#' * 30}\n{model_response_content.strip()}\n{'#' * 30}\n"
        logging.info(judgment_text)

    def judge_video(self, summaries: list[str]) -> list[float]:
        """
        Judges a list of video summaries for anomalies using the Gemma model,
        and returns a list of extracted anomaly scores.
        """
        logging.info(f"Starting anomaly detection for {len(summaries)} summaries.")

        self.clear_context()

        anomaly_scores = []

        for idx, summary in enumerate(summaries):
            for prompt_id, judge_prompt in self.judge_prompts.items():
                _system_prompt = self.prefix_prompt + judge_prompt + self.system_prompt
                full_prompt = f"Frame Description:\n{summary}\n\nResponse:"

                logging.info(f"Processing summary chunk {idx} for anomaly judgment...")
                response_content = self._generate_text(_system_prompt, full_prompt)
                logging.debug(f"Raw model response for chunk {idx}: {response_content}")
                model_response_content = prompt_id + ": " + response_content

                self._display_judgment(idx, model_response_content)

            self._master_judge()

        logging.info("Anomaly detection finished for all summaries.")
        return anomaly_scores
