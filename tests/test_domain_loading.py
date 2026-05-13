from pathlib import Path

import pytest

from autosynth.domain import build_domain, get_domain_class, load_domain_from_path


def test_get_registered():
    cls = get_domain_class("qa_from_documents")
    assert cls.__name__ == "QAFromDocuments"


def test_unknown_raises():
    with pytest.raises(KeyError):
        get_domain_class("does_not_exist")


def test_load_from_path(tmp_path: Path):
    src = tmp_path / "tiny.py"
    src.write_text(
        "from autosynth.domain import DomainAdapter\n"
        "class Tiny(DomainAdapter):\n"
        "    def load_grounding(self): return []\n"
        "    def generation_prompt(self, *a, **k): return []\n"
        "    def validate_candidate(self, c): return []\n"
        "    def solver_prompt(self, c, r): return []\n"
        "    def quality_prompt(self, c): return []\n"
        "    def judge_prompt(self, c, r, s): return []\n"
    )
    cls = load_domain_from_path(f"{src}:Tiny")
    assert cls.__name__ == "Tiny"


def test_build_domain_with_params(sample_docs: Path):
    d = build_domain("qa_from_documents", None, {"source_dir": str(sample_docs)})
    items = list(d.load_grounding())
    assert len(items) == 2
    assert all(i.body for i in items)
