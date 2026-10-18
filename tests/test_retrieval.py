from helpdesk_copilot import config
from helpdesk_copilot.retrieval import KnowledgeBase, LocalHashEmbedder, load_passages


def test_every_section_becomes_a_passage_with_a_stable_id():
    passages = load_passages(config.KB_DIR)
    ids = [p.id for p in passages]
    assert len(ids) == len(set(ids)) >= 15
    assert "refund-policy#1" in ids
    assert all(p.text for p in passages)


def test_search_finds_the_right_policy():
    kb = KnowledgeBase(config.KB_DIR, LocalHashEmbedder())
    top = kb.search("I bought the annual plan 10 days ago, can I get a refund?", k=3)
    assert top[0].id == "refund-policy#1"
    assert top[0].score >= top[-1].score


def test_search_finds_password_reset():
    kb = KnowledgeBase(config.KB_DIR, LocalHashEmbedder())
    top = kb.search("I forgot my password and the reset link expired", k=2)
    assert top[0].source == "password-reset-sop.md"


def test_unrelated_query_scores_low():
    kb = KnowledgeBase(config.KB_DIR, LocalHashEmbedder())
    related = kb.search("refund for annual plan", k=1)[0].score
    unrelated = kb.search("what is the weather in Paris tomorrow", k=1)[0].score
    assert unrelated < related
