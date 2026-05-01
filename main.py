"""Entry point.

No args  →  launch the Textual TUI (`hf_agent.app.ChatApp`).
With args →  run one NL query headless and print the transcript.
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="hf-agent",
        description="Natural-language → Hugging Face Hub / Datasets Server query agent.",
    )
    parser.add_argument(
        "query", nargs="*",
        help="If given, run one NL query headless instead of launching the TUI.",
    )
    parser.add_argument(
        "--model",
        help="Model to use, e.g. 'gemini-2.0-flash' or 'gemini:gemini-2.0-flash'. "
             "Defaults to the first configured model.",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=6,
        help="Max tool-loop rounds before stopping (default 6).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit NDJSON events instead of human-readable text.",
    )
    args = parser.parse_args()

    if args.query:
        from hf_agent.headless import cli_run
        query = " ".join(args.query)
        sys.exit(cli_run(
            query,
            model=args.model,
            max_rounds=args.max_rounds,
            as_json=args.json,
        ))

    from hf_agent import ChatApp
    ChatApp().run()


if __name__ == "__main__":
    main()
