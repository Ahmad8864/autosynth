from autodata.config import ModelConfig
from autodata.models import LLMClient


def test_mock_provider_basic():
    client = LLMClient(ModelConfig(provider_model="mock/happy"), role="weak")
    resp = client.complete([{"role": "user", "content": "ROLE:WEAK_SOLVER question?"}])
    assert "vague" in resp.text


def test_mock_provider_json():
    client = LLMClient(ModelConfig(provider_model="mock/happy"), role="judge")
    data = client.complete_json([{"role": "user", "content": "ROLE:JUDGE [solver=strong] anything"}])
    assert data["total"] > 0.8
    assert "per_criterion" in data
