from helpdesk_copilot.schemas import Claim, ClaimVerdict, Draft, Judgement
from helpdesk_copilot.validation import check_response

PASSAGES = [
    {"id": "refund-policy#1", "text": "Customers may request a refund within 30 days of the original purchase date.", "score": 0.8},
    {"id": "refund-policy#3", "text": "Approved refunds are returned within 5 to 10 business days.", "score": 0.5},
]
TICKET = "I bought the annual plan 10 days ago. Can I get a refund?"


def good_draft(**over) -> Draft:
    base = dict(
        reply="You can request a refund within 30 days of purchase.",
        claims=[Claim(text="Refunds can be requested within 30 days of purchase.", source_ids=["refund-policy#1"])],
        needs_human=False,
    )
    base.update(over)
    return Draft(**base)


def supported(n=1) -> Judgement:
    return Judgement(verdicts=[ClaimVerdict(claim_index=i, supported=True, note="ok") for i in range(1, n + 1)])


def run(draft, judgement, passages=PASSAGES, ticket=TICKET, min_score=0.3):
    return check_response(draft, judgement, passages, ticket, min_score)


def test_clean_reply_passes():
    assert run(good_draft(), supported()) == []


def test_unknown_source_is_rejected():
    d = good_draft(claims=[Claim(text="x", source_ids=["made-up#9"])])
    assert any("unknown source" in r for r in run(d, supported()))


def test_claim_without_source_is_rejected():
    d = good_draft(claims=[Claim(text="x", source_ids=[])])
    assert any("has no source" in r for r in run(d, supported()))


def test_judge_marks_claim_unsupported():
    j = Judgement(verdicts=[ClaimVerdict(claim_index=1, supported=False, note="adds a detail")])
    assert any("not supported" in r for r in run(good_draft(), j))


def test_unjudged_claim_is_rejected():
    assert any("not judged" in r for r in run(good_draft(), Judgement(verdicts=[])))


def test_missing_judgement_is_rejected():
    assert any("no grounding judgement" in r for r in run(good_draft(), None))


def test_invented_number_is_rejected():
    d = good_draft(reply="You can request a refund within 45 days of purchase.")
    assert any("45" in r for r in run(d, supported()))


def test_number_from_the_ticket_is_allowed():
    d = good_draft(reply="Since you bought it 10 days ago, you are within 30 days.")
    assert run(d, supported()) == []


def test_refund_promise_is_blocked():
    d = good_draft(reply="Your refund will be refunded in 30 days.")
    assert any("refund promise" in r for r in run(d, supported()))


def test_guarantee_is_blocked():
    d = good_draft(reply="We guarantee a refund within 30 days.")
    assert any("guarantee" in r for r in run(d, supported()))


def test_weak_retrieval_escalates():
    weak = [{**p, "score": 0.1} for p in PASSAGES]
    assert any("weak retrieval" in r for r in run(good_draft(), supported(), passages=weak))


def test_needs_human_escalates():
    assert any("flagged" in r for r in run(good_draft(needs_human=True), supported()))


def test_no_claims_escalates():
    assert any("no cited claims" in r for r in run(good_draft(claims=[]), None))
