"""Make the sibling ``support_agent`` package importable without installing it."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT.parent / "01-agentic-support-assistant" / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))
