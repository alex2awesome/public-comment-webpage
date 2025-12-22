from training.rollout_openenv import _parse_action_payload


def test_parse_action_payload_normalizes_underscored_type():
    raw = (
        '<think>thoughts</think>\n'
        '<action>{"type":"SEARCH SEMANTIC_SCHOLAR","query":"state carbon pricing"}</action>'
    )
    action, thought = _parse_action_payload(raw)
    assert action["type"] == "SEARCH SEMANTIC SCHOLAR"
    assert thought == "thoughts"


def test_parse_action_payload_handles_legacy_alias():
    raw = '<action>{"type":"fetch_paper","paper_id":"12345"}</action>'
    action, _ = _parse_action_payload(raw)
    assert action["type"] == "FETCH PAPER"
    assert action["paper_id"] == "12345"
