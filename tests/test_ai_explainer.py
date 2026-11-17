"""AI layer: must work without a key, survive API failures, and never leak identifiers."""
import json
from types import SimpleNamespace

import pytest

from config import ReconConfig
from data.generate_data import generate_datasets
from reconciliation import ai_explainer
from reconciliation.reconciliation_engine import run_reconciliation


@pytest.fixture(scope="module")
def result():
    data = generate_datasets(42)
    return run_reconciliation(data.platform, data.bank, ReconConfig())


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch):
    monkeypatch.setattr(ai_explainer, "get_groq_api_key", lambda: None)


class FakeClient:
    """Mimics client.chat.completions.create(...)."""

    def __init__(self, text=None, error=None):
        self._text, self._error = text, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.last_messages = None

    def _create(self, **kwargs):
        self.last_messages = kwargs["messages"]
        if self._error:
            raise self._error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))])


def test_without_api_key_a_labelled_template_summary_is_returned(result):
    report = ai_explainer.generate_summary(result)
    assert not report.is_ai and report.source == ai_explainer.SOURCE_TEMPLATE
    assert "GROQ_API_KEY" in report.error
    assert "Month-end reconciliation summary" in report.text
    assert result.summary["completion_status"] in report.text
    assert "Duplicate Entry" in report.text and "Unmatched Refund" in report.text


def test_ai_text_is_labelled_as_ai_and_checked_against_engine_numbers(result):
    n = result.summary["total_platform_records"]
    client = FakeClient(text=f"The engine compared {n} platform records.")
    report = ai_explainer.generate_summary(result, client=client)
    assert report.is_ai and report.source == ai_explainer.SOURCE_AI
    assert report.unverified_numbers == []


def test_invented_numbers_are_flagged(result):
    client = FakeClient(text="A total of 987654 transactions failed.")
    report = ai_explainer.generate_summary(result, client=client)
    assert report.is_ai and report.unverified_numbers == ["987654"]


def test_api_failure_falls_back_to_template_and_hides_the_key(result):
    client = FakeClient(error=TimeoutError("request to Groq timed out using key gsk-secret-123"))
    report = ai_explainer.generate_summary(result, api_key="gsk-secret-123", client=client)
    assert not report.is_ai and "AI request failed" in report.error
    assert "gsk-secret-123" not in report.error and "TimeoutError" in report.error
    assert "Month-end reconciliation summary" in report.text


def test_empty_model_response_falls_back(result):
    report = ai_explainer.generate_summary(result, client=FakeClient(text="   "))
    assert not report.is_ai and "empty response" in report.error


def test_data_sent_to_the_model_has_no_customer_or_bank_identifiers(result):
    client = FakeClient(text="ok")
    ai_explainer.generate_summary(result, client=client)
    sent = json.dumps(client.last_messages)
    assert "CUST-" not in sent and "BNK" not in sent and "STL5" not in sent
    context = json.dumps(ai_explainer.build_llm_context(result), default=str)
    assert "CUST-" not in context and "BNK" not in context


def test_explain_one_discrepancy_uses_the_category_template_as_fallback(result):
    row = result.discrepancies[result.discrepancies["category"] == "Duplicate Entry"].iloc[0].to_dict()
    report = ai_explainer.explain_discrepancy(result, row)
    assert not report.is_ai
    assert "Possible causes" in report.text and "idempotency" in report.text
    assert row["details"] in report.text


def test_find_unverified_numbers_ignores_small_numbers_and_known_values():
    context = {"counts": {"matched": 104}, "amount": "7804.31"}
    assert ai_explainer.find_unverified_numbers("104 matched, 7804.31 gap, 3 steps, 1-2 days", context) == []
    assert ai_explainer.find_unverified_numbers("There were 105 matched", context) == ["105"]
