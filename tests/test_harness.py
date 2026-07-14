from autosynth.harness import DEFAULT_HARNESS, apply_harness, make_harness


def test_default_harness_has_seed_rules():
    assert DEFAULT_HARNESS.challenger_rules
    assert DEFAULT_HARNESS.quality_rules
    assert DEFAULT_HARNESS.judge_rules
    assert DEFAULT_HARNESS.rubric_max_weight == 7


def test_solver_rules_are_shared():
    h = make_harness(solver_rules=["show work"])
    assert h.rules_for("solver") == ["show work"]


def test_apply_harness_appends_to_system_message():
    msgs = [
        {"role": "system", "content": "base instructions"},
        {"role": "user", "content": "user content"},
    ]
    out = apply_harness(msgs, ["rule one", "rule two"])
    assert "ADDITIONAL RULES" in out[0]["content"]
    assert "rule one" in out[0]["content"]
    assert "rule two" in out[0]["content"]
    assert out[1] == msgs[1]


def test_apply_harness_with_no_system_message_prepends_one():
    msgs = [{"role": "user", "content": "hello"}]
    out = apply_harness(msgs, ["rule"])
    assert out[0]["role"] == "system"
    assert "rule" in out[0]["content"]
    assert out[1]["role"] == "user"


def test_apply_harness_noop_for_empty_rules():
    msgs = [{"role": "system", "content": "x"}]
    assert apply_harness(msgs, []) == msgs


def test_self_test_rule_only_when_enabled():
    h = make_harness(challenger_rules=["a"], require_self_test=False)
    rules = h.rules_for("challenger")
    assert rules == ["a"]
    h2 = make_harness(challenger_rules=["a"], require_self_test=True)
    rules2 = h2.rules_for("challenger")
    assert len(rules2) == 2
    assert "self_test" in rules2[1]


def test_fingerprint_ignores_iteration_and_scores():
    h1 = make_harness(challenger_rules=["x"], iteration=0)
    h2 = make_harness(challenger_rules=["x"], iteration=99)
    h2.train_score = 0.5
    h2.val_score = 0.7
    assert h1.fingerprint() == h2.fingerprint()
    h3 = make_harness(challenger_rules=["y"])
    assert h1.fingerprint() != h3.fingerprint()
