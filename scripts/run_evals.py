"""Run the ADK evals with the judge model taken from .env (EVAL_JUDGE_MODEL).

    python scripts/run_evals.py

ADK's own default judge model (gemini-2.5-flash) is closed to new API users, and
evals/test_config.json cannot read environment variables. This script copies that
file, sets the judge model from EVAL_JUDGE_MODEL (default: JUDGE_MODEL) and then runs
`adk eval` for you. Nothing else is changed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from helpdesk_copilot import config  # noqa: E402


def render_config(template: dict, judge_model: str) -> dict:
    """Pure function: return a copy of the eval config with the judge model set."""
    cfg = json.loads(json.dumps(template))
    for criterion in cfg.get("criteria", {}).values():
        if isinstance(criterion, dict):  # criteria given as a bare number have no judge
            criterion.setdefault("judge_model_options", {})["judge_model"] = judge_model
    return cfg


def main() -> int:
    template = json.loads((ROOT / "evals" / "test_config.json").read_text(encoding="utf-8"))
    cfg = render_config(template, config.EVAL_JUDGE_MODEL)
    config.CACHE_DIR.mkdir(exist_ok=True)
    out = config.CACHE_DIR / "eval_config.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    adk = shutil.which("adk", path=str(Path(sys.executable).parent)) or shutil.which("adk")
    if not adk:
        print("Could not find the `adk` command. Activate your virtual environment first.")
        return 1
    print(f"Judge model: {config.EVAL_JUDGE_MODEL}  (change it with EVAL_JUDGE_MODEL in .env)")
    cmd = [
        adk, "eval", "helpdesk_copilot", str(ROOT / "evals" / "helpdesk.evalset.json"),
        f"--config_file_path={out}", "--print_detailed_results",
    ]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
