import logging
import re
import torch
import torch._dynamo
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import os
from hf_auth import get_hf_token

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
torch.backends.cudnn.enabled = False
torch.set_float32_matmul_precision('high')
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

        Args:
            model_name (str): The name of the HuggingFace Gemma 3 model to use.
            device (str): Device to run the model on ('auto', 'cuda', 'cpu').
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        hf_token = get_hf_token()
        
        self.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            attn_implementation="eager",
            token=hf_token
        ).eval()
        
        self.task_prompt = (
            "Rate the anomaly level in this video scene description on a scale from 0.0 to 1.0, "
            "where 0.0 means completely normal and 1.0 means highly anomalous. "
            "Provide your score as a decimal number (e.g., 0.3 or 0.8). "
            "Score: "
        )
        self.system_prompts = {
                    "Judge": "Analyze this video scene description for any unusual or anomalous behavior.",
                }
        
        logging.info(f"AnomalyDetector initialized with model: {self.model_name}, device: {self.device}")

    def _extract_score_from_response(self, model_response_content: str) -> float:
        """
        Extracts the anomaly score from the LLM's raw text response.

        Args:
            model_response_content (str): The raw text content received from the LLM.

        Returns:
            float: The extracted anomaly score, clamped between 0.0 and 1.0, or -1.0 if parsing fails.
        """
        score_tag_match = re.search(r'</?SCORE>([0-9]*\.?[0-9]+)', model_response_content, re.IGNORECASE)
        if score_tag_match:
            try:
                score = float(score_tag_match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        score_match = re.search(r'([0-9]*\.?[0-9]+)', model_response_content)
        if score_match:
            try:
                score = float(score_match.group(1))
                if score > 1.0:
                    score = score / 100.0
                return max(0.0, min(1.0, score))
            except ValueError:
                logging.warning(f"Could not convert extracted score to float: '{score_match.group(1)}'")
                return 0.1
        else:
            logging.warning(f"No valid float score found in LLM response. Response: '{model_response_content.strip()}'")
            return 0.1

    def _generate_text(self, prompt: str, max_length: int = 512) -> str:
        """
        Generates text using the Gemma 3 model.

        Args:
            prompt (str): The input prompt for text generation.
            max_length (int): Maximum length of generated text.

        Returns:
            str: Generated text response.
        """
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]
        
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()

    def _display_judgment(self, idx: int, model_response_content: str):
        """
        Prints the LLM's judgment to the console.

        Args:
            idx (int): The index of the summary chunk being judged.
            model_response_content (str): The content of the LLM's response.
        """
        print("\n" + "#"*30)
        print(f" ANOMALY JUDGMENT (SUMMARY CHUNK {idx})")
        print("#"*30)
        print(model_response_content.strip())
        print("#"*30 + "\n")

    def judge_video(self, summaries: list[str]) -> list[float]:
        """
        Judges a list of video summaries for anomalies using the Gemma model,
        and returns a list of extracted anomaly scores.

        Args:
            summaries (list[str]): A list of summarized text descriptions (e.g., chunked summaries).

        Returns:
            list[float]: A list of anomaly scores (floats between 0 and 1) extracted from model responses.
        """
        logging.info(f"Starting anomaly detection for {len(summaries)} summaries.")
        
        anomaly_scores = []
        
        for idx, summary in enumerate(summaries):
            chunk_score = 0
            for prompt_id, system_prompt in self.system_prompts.items():
                full_prompt = f"{system_prompt}\n\n{self.task_prompt}\n\nFrame Description:\n{summary}\n\nResponse:"
                
                logging.info(f"Processing summary chunk {idx} for anomaly judgment...")
                try:
                    response_content = self._generate_text(full_prompt)
                    logging.debug(f"Raw model response for chunk {idx}: {response_content}")
                    model_response_content = prompt_id + ": " + response_content
                
                    self._display_judgment(idx, model_response_content)
                
                    current_score = self._extract_score_from_response(response_content)
                    chunk_score += current_score
                    logging.info(f"Extracted score for chunk {idx}: {current_score}")
                except Exception as e:
                    logging.error(f"Error processing chunk {idx}: {e}")
                    current_score = 0.1  # Use small default instead of 0
                    chunk_score += current_score
            
            chunk_score /= len(self.system_prompts)
            logging.info(f"Average chunk score: {chunk_score}")
            anomaly_scores.append(chunk_score)
            
        logging.info("Anomaly detection finished for all summaries.")
        return anomaly_scores
