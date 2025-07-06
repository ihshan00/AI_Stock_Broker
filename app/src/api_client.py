import requests
from .config import Config

class ChatAPIClient:
    """
    Encapsulates communication with the chatbot backend.
    """

    def __init__(self):
        if not Config.BACKEND_URL:
            raise ValueError("BACKEND_URL is not set in environment.")
        self.url = Config.BACKEND_URL

    def send_message(self, message: str, history: list[tuple[str, str | None]]):
        payload = {"prompt": message}  # Only send 'prompt'
        resp = requests.post(self.url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        bot_reply = data.get("response") or "No response from backend."
        return message, bot_reply

