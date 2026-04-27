from dotenv import load_dotenv

from hf_agent import ChatApp


def main() -> None:
    load_dotenv()
    ChatApp().run()


if __name__ == "__main__":
    main()
