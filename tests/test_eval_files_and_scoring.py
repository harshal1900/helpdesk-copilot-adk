import csv
import json
import sys
from pathlib import Path

import pytest
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_set import EvalSet

from helpdesk_copilot.schemas import Triage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from measure_routing import score  # noqa: E402


def test_evalset_matches_adk_schema():
    eval_set = EvalSet.model_validate_json((ROOT / "evals" / "helpdesk.evalset.json").read_text())
    assert len(eval_set.eval_cases) == 3


def test_eval_config_matches_adk_schema():
    cfg = EvalConfig.model_validate(json.loads((ROOT / "evals" / "test_config.json").read_text()))
    assert "final_response_match_v2" in cfg.criteria


def test_labelled_tickets_use_valid_labels():
    with open(ROOT / "data" / "labelled_tickets.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 21
    for r in rows:  # every label must be a value the Triage schema accepts
        Triage(queue=r["queue"], ticket_type=r["ticket_type"], priority=r["priority"], confidence=1.0, rationale="x")


def _pred(queue="billing", t="question", p="P3"):
    return Triage(queue=queue, ticket_type=t, priority=p, confidence=0.9, rationale="x")


def test_score_perfect_and_partial():
    rows = [
        {"queue": "billing", "ticket_type": "question", "priority": "P3"},
        {"queue": "account", "ticket_type": "request", "priority": "P3"},
    ]
    perfect = score(rows, [_pred(), _pred("account", "request", "P3")])
    assert perfect["exact_match_accuracy"] == 1.0

    partial = score(rows, [_pred(), _pred("technical", "request", "P1")])
    assert partial["queue_accuracy"] == 0.5
    assert partial["ticket_type_accuracy"] == 1.0
    assert partial["priority_accuracy"] == 0.5
    assert partial["exact_match_accuracy"] == 0.5
    assert partial["queue_confusions"] == {("account", "technical"): 1}


def test_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        score([{"queue": "billing", "ticket_type": "question", "priority": "P3"}], [])
