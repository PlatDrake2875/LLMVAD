import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import HfApi, login, logout


class HuggingFaceAuth:
    """
    Enhanced HuggingFace authentication with caching and interactive login.
    """

    def __init__(self, cache_dir: str = ".hf_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.token_cache_file = self.cache_dir / "token_cache.json"
        self.auth_cache_file = self.cache_dir / "auth_cache.json"

    def _load_cached_token(self) -> Optional[str]:
        """Load cached token if it exists and is valid."""
        if not self.token_cache_file.exists():
            return None

        try:
            with open(self.token_cache_file, "r") as f:
                cache_data = json.load(f)

            if cache_data.get("token") and cache_data.get("timestamp"):
                return cache_data["token"]
        except Exception as e:
            logging.warning(f"Failed to load cached token: {e}")

        return None

    def _save_token_cache(self, token: str):
        """Save token to cache."""
        try:
            cache_data = {
                "token": token,
                "timestamp": str(Path().stat().st_mtime),
            }
            with open(self.token_cache_file, "w") as f:
                json.dump(cache_data, f)
            logging.info("Token cached successfully")
        except Exception as e:
            logging.warning(f"Failed to cache token: {e}")

    def _clear_token_cache(self):
        """Clear cached token."""
        try:
            if self.token_cache_file.exists():
                self.token_cache_file.unlink()
            logging.info("Token cache cleared")
        except Exception as e:
            logging.warning(f"Failed to clear token cache: {e}")

    def _validate_token(self, token: str) -> bool:
        """Validate if the token is working."""
        try:
            api = HfApi(token=token)
            user_info = api.whoami()
            logging.info(f"Token validated for user: {user_info['name']}")
            return True
        except Exception as e:
            logging.warning(f"Token validation failed: {e}")
            return False

    def _interactive_login(self) -> Optional[str]:
        """Interactive login via command line."""
        print("\n" + "=" * 50)
        print("HuggingFace Authentication Required")
        print("=" * 50)
        print("To access the Gemma model, you need to:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'read' access")
        print(
            "3. Accept the model terms at https://huggingface.co/google/gemma-3-4b-it"
        )
        print("=" * 50)

        while True:
            token = input(
                "\nEnter your HuggingFace API token (or 'quit' to exit): "
            ).strip()

            if token.lower() == "quit":
                return None

            if not token:
                print("Token cannot be empty. Please try again.")
                continue

            # Validate the token
            if self._validate_token(token):
                return token
            else:
                print("Invalid token. Please check your token and try again.")

    def authenticate(self, force_interactive: bool = False) -> Optional[str]:
        """
        Authenticate with HuggingFace using cached token or interactive login.

        Args:
            force_interactive: Force interactive login even if cached token exists

        Returns:
            Valid token or None if authentication failed
        """
        load_dotenv()

        env_token = os.getenv("HUGGINGFACE_API_KEY")
        if env_token and env_token != "your_hf_api_key_here":
            if self._validate_token(env_token):
                self._save_token_cache(env_token)
                return env_token
            else:
                logging.warning("Environment token is invalid")

        if not force_interactive:
            cached_token = self._load_cached_token()
            if cached_token and self._validate_token(cached_token):
                logging.info("Using cached authentication token")
                return cached_token
            elif cached_token:
                logging.warning("Cached token is invalid, clearing cache")
                self._clear_token_cache()

        token = self._interactive_login()
        if token:
            self._save_token_cache(token)
            return token

        return None

    def setup_auth(self) -> bool:
        """
        Set up HuggingFace authentication and login.

        Returns:
            True if authentication successful, False otherwise
        """
        token = self.authenticate()

        if not token:
            logging.error("Failed to authenticate with HuggingFace")
            return False

        try:
            login(token=token, add_to_git_credential=False)
            logging.info("Successfully logged in to HuggingFace")
            return True

        except Exception as e:
            logging.error(f"Failed to login to HuggingFace: {e}")
            return False

    def logout(self):
        """Logout from HuggingFace and clear cache."""
        try:
            logout()
            self._clear_token_cache()
            logging.info("Logged out from HuggingFace and cleared cache")
        except Exception as e:
            logging.warning(f"Failed to logout: {e}")


# Global instance for convenience
_hf_auth = HuggingFaceAuth()


def setup_huggingface_auth() -> bool:
    """
    Sets up HuggingFace authentication using API key from .env file or interactive login.
    """
    return _hf_auth.setup_auth()


def get_hf_token() -> Optional[str]:
    """
    Gets the HuggingFace API token from environment variables or cache.
    """
    return _hf_auth.authenticate()


def interactive_login() -> Optional[str]:
    """
    Force interactive login to HuggingFace.
    """
    return _hf_auth.authenticate(force_interactive=True)


def logout_hf():
    """
    Logout from HuggingFace and clear cache.
    """
    _hf_auth.logout()


def main():
    """Command-line interface for managing HuggingFace authentication."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage HuggingFace authentication")
    parser.add_argument(
        "action",
        choices=["login", "logout", "status", "test"],
        help="Action to perform",
    )
    parser.add_argument(
        "--force-interactive",
        action="store_true",
        help="Force interactive login even if cached token exists",
    )

    args = parser.parse_args()

    if args.action == "login":
        if args.force_interactive:
            print("Forcing interactive login...")
            token = interactive_login()
            if token:
                print("✓ Interactive login successful!")
            else:
                print("✗ Interactive login failed or cancelled")
        else:
            print("Attempting login with cached token or environment variable...")
            success = setup_huggingface_auth()
            if success:
                print("✓ Login successful!")
            else:
                print("✗ Login failed")

    elif args.action == "logout":
        print("Logging out and clearing cache...")
        logout_hf()
        print("✓ Logout complete")

    elif args.action == "status":
        print("Checking authentication status...")

        # Check environment variable
        env_token = os.getenv("HUGGINGFACE_API_KEY")
        if env_token and env_token != "your_hf_api_key_here":
            print(f"✓ Environment variable set: {env_token[:10]}...")
        else:
            print("✗ Environment variable not set")

        # Check cached token
        token = get_hf_token()
        if token:
            print("✓ Cached token available")
        else:
            print("✗ No cached token")

    elif args.action == "test":
        print("Testing authentication...")
        token = get_hf_token()
        if token:
            print("✓ Authentication test passed")
            print(f"Token: {token[:10]}...")
        else:
            print("✗ Authentication test failed")


if __name__ == "__main__":
    main()
