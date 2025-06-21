import json
import requests
import logging
import re # Import regex for parsing scores

class AnomalyDetector():
    def __init__(self, api_url: str = "http://localhost:11434/api/chat", timeout: int = 30, model_name: str = "gemma3:4b-it-q4_K_M"):
        self.api_url = api_url
        self.task_prompt = (
            "You are given video frames descriptions, one at a time. Output an anomaly score from 0 to 1 and give your reasoning. "
            "Output your response in this format:\n<REASONING>.\n<SCORE>."
        )
        self.system_prompt = "You're a police officer investigating the likelihood of someone having broken the law." 
        self.model_name = model_name
        self.timeout = timeout

        logging.info(f"AnomalyDetector initialized with API URL: {self.api_url}, model: {self.model_name}")

    def judge_video(self, summaries: list[str]) -> list[float]:
        """
        Judges a list of video summaries for anomalies using rolling context with the LLM,
        and returns a list of extracted anomaly scores.

        Args:
            summaries (list[str]): A list of summarized text descriptions (e.g., chunked summaries).

        Returns:
            list[float]: A list of anomaly scores (floats between 0 and 1) extracted from LLM responses.
        """
        logging.info(f"Starting anomaly detection for {len(summaries)} summaries with rolling context.")
        
        # Initialize chat history with the system prompt
        chat_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        anomaly_scores = []
        
        for idx, summary in enumerate(summaries):
            # Formulate the user's turn including the task prompt and the current summary
            user_message_content = f"{self.task_prompt}\n\nFrame Description:\n{summary}"
            
            # Add the user's current message to the history
            chat_history.append({"role": "user", "content": user_message_content})
            
            payload = {
                "model": self.model_name,
                "messages": chat_history,
                "stream": False 
            }

            logging.info(f"Sending summary chunk {idx} for anomaly judgment with context...")
            current_score = -1.0 # Default score if parsing fails

            try:
                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                response.raise_for_status()

                response_data = response.json()
                model_response_content = response_data.get("message", {}).get("content", "No score was given.")
                
                # --- Extract score using the corrected regex ---
                # This regex now expects '<SCORE>.' followed by optional whitespace,
                # then captures the float number, without requiring a closing tag.
                score_match = re.search(r'<SCORE>\.\s*(\d*\.?\d+)', model_response_content, re.IGNORECASE)
                
                if score_match:
                    try:
                        current_score = float(score_match.group(1))
                        # Ensure score is within 0 and 1
                        current_score = max(0.0, min(1.0, current_score)) 
                        logging.info(f"Extracted score for chunk {idx}: {current_score}")
                    except ValueError:
                        logging.warning(f"Could not convert extracted score to float for chunk {idx}: '{score_match.group(1)}'")
                else:
                    logging.warning(f"No <SCORE>. followed by a number found in LLM response for chunk {idx}. Response: '{model_response_content.strip()}'")
                # --- End score extraction ---

                # Add the LLM's response to the chat history for the next turn
                chat_history.append({"role": "assistant", "content": model_response_content})

                print("\n" + "#"*30)
                print(f" ANOMALY JUDGMENT (SUMMARY CHUNK {idx})")
                print("#"*30)
                print(model_response_content.strip())
                print("#"*30 + "\n")

            except requests.exceptions.RequestException as e:
                logging.error(f"Error judging summary chunk {idx}: {e}")
                print(f"\nError: Could not get anomaly judgment for chunk {idx}. {e}\n")
                # Add an error message to history to maintain structure, and a placeholder score
                chat_history.append({"role": "assistant", "content": f"Error processing this chunk: {e}"})
                current_score = -1.0 # Indicate an error in score

            anomaly_scores.append(current_score) # Always append a score, even if default/error
            
        logging.info("Anomaly detection finished for all summaries.")
        return anomaly_scores

