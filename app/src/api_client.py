import requests
from ..config import Config

class ChatAPIClient:
    """
    Encapsulates communication with the chatbot backend.
    """

    def __init__(self):
        if not Config.BACKEND_URL:
            raise ValueError("BACKEND_URL is not set in environment.")
        self.url = Config.BACKEND_URL

    def send_message(self, message: str, history: list[tuple[str, str | None]]) -> tuple[str, str]:
        """
        Sends `message` and `history` to the backend, returns (user_msg, bot_reply).
        """
        payload = {"message": message, "history": history}
        resp = requests.post(self.url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        bot_reply = data.get("reply") or "I couldn't process that."
        return message, bot_reply
