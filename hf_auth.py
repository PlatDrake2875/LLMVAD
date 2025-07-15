import os
import logging
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from typing import Optional


def setup_huggingface_auth() -> bool:
    """
    Sets up HuggingFace authentication using API key from .env file.
    
    Returns:
        bool: True if authentication was successful, False otherwise.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get HuggingFace API key from environment
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not hf_api_key or hf_api_key == "your_hf_api_key_here":
        logging.warning(
            "HuggingFace API key not found or not set in .env file. "
            "Some models may not be accessible. Please set HUGGINGFACE_API_KEY in .env file."
        )
        return False
    
    try:
        # Login to HuggingFace Hub
        login(token=hf_api_key, add_to_git_credential=False)
        
        # Verify authentication by checking user info
        api = HfApi()
        user_info = api.whoami()
        
        logging.info(f"Successfully authenticated with HuggingFace as user: {user_info['name']}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to authenticate with HuggingFace: {e}")
        logging.warning("Continuing without HuggingFace authentication. Some models may not be accessible.")
        return False


def get_hf_token() -> Optional[str]:
    """
    Gets the HuggingFace API token from environment variables.
    
    Returns:
        str or None: The HuggingFace API token if available, None otherwise.
    """
    load_dotenv()
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not hf_api_key or hf_api_key == "your_hf_api_key_here":
        return None
    
    return hf_api_key
