"""Safety filtering of candidate payloads and reference answers."""

from __future__ import annotations

from autosynth.pipeline import _safety_text
from autosynth.safety import default_filter
from autosynth.schemas import Candidate


def _cand(payload: dict, reference: str | None = None) -> Candidate:
    return Candidate(candidate_id="c", domain="d", source_id="s", payload=payload, reference_output=reference)


def test_safety_text_includes_nested_and_reference():
    text = _safety_text(
        _cand(
            {"q": "outer", "meta": {"note": "nested-value"}, "tags": ["listed"], "n": 7},
            reference="the answer",
        )
    )
    for token in ("outer", "nested-value", "listed", "7", "the answer"):
        assert token in text


def test_pii_in_reference_output_detected():
    verdict = default_filter(_safety_text(_cand({"q": "what is the ssn?"}, reference="It is 123-45-6789")))
    assert not verdict.allowed and any("SSN" in r for r in verdict.reasons)


def test_pii_in_nested_payload_detected():
    verdict = default_filter(_safety_text(_cand({"q": "q", "ctx": {"contact": "alice@example.com"}})))
    assert not verdict.allowed and any("email" in r for r in verdict.reasons)


def test_clean_candidate_allowed():
    assert default_filter(_safety_text(_cand({"q": "what is 2+2?"}, reference="four"))).allowed
