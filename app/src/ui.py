import gradio as gr
from .api_client import ChatAPIClient


def build_interface() -> gr.Blocks:
    """
    Builds and returns the Gradio Blocks interface with chatbot and controls.
    """
    client = ChatAPIClient()

    def handle(message, history):
        history = history or []
        user_msg, bot_msg = client.send_message(message, history)
        history.append((user_msg, bot_msg))
        return history, history

    def clear():
        return [], []

    with gr.Blocks(title="AI Stock Broker") as demo:
        chatbot = gr.Chatbot(label="Assistant")
        state = gr.State([])

        with gr.Row():
            text = gr.Textbox(
                placeholder="Type a message...",
                show_label=False,
                lines=1,
                container=True
            )
            send = gr.Button("Send")

        send.click(handle, inputs=[text, state], outputs=[chatbot, state])
        text.submit(handle, inputs=[text, state], outputs=[chatbot, state])

        clear_btn = gr.Button("Clear History")
        clear_btn.click(clear, inputs=None, outputs=[chatbot, state])

    return demo
