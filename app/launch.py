from app.ui import build_interface


def main():
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False
    )


if __name__ == "__main__":
    main()
