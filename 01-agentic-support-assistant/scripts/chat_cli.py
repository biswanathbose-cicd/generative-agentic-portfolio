"""Interactive terminal chat:  python scripts/chat_cli.py [--version v1|v2] [--trace]"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent import Session, V1_BASELINE, V2_IMPROVED, build_assistant  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--trace", action="store_true", help="print tool calls")
    args = ap.parse_args()
    assistant = build_assistant(V1_BASELINE if args.version == "v1" else V2_IMPROVED)
    session = Session()
    print("Support assistant. Try: 'where is my order WM-10003?'  (Ctrl-D to quit)")
    while True:
        try:
            text = input("you> ")
        except EOFError:
            break
        r = assistant.handle(session, text)
        print(f"bot> {r.response}")
        if args.trace:
            print(f"     intent={r.intent} outcome={r.outcome} flags={r.guardrail_flags}")
            for t in r.trace:
                print(f"     tool {t['tool']}({t['args']})")


if __name__ == "__main__":
    main()
