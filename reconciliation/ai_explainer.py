"""AI explanations for reconciliation results.

Design rules:

* The deterministic engine finds every discrepancy. The LLM only *explains*
  results the engine already produced and never changes them.
* Only aggregated, anonymised data goes to the model: no customer IDs, bank
  references or settlement IDs.
* If there is no API key, or the API fails/times out, the app falls back to
  template-based text so it keeps working. Output is always labelled with its source.
* Numbers in the LLM text are cross-checked against the engine output and any
  figure that cannot be found there is flagged for the reader.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from config import (
    CAT_AMOUNT_MISMATCH,
    CAT_DUPLICATE,
    CAT_MISSING_BANK,
    CAT_MISSING_PLATFORM,
    CAT_REFUND_INCONSISTENT,
    CAT_ROUNDING,
    CAT_STATUS_CONFLICT,
    CAT_UNMATCHED_REFUND,
    GROQ_MAX_RETRIES,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
    get_groq_api_key,
)
from reconciliation.reconciliation_engine import ReconciliationResult

SOURCE_AI = "AI-generated (Groq)"
SOURCE_TEMPLATE = "Template-based (no AI used)"

SYSTEM_PROMPT = (
    "You are a payments operations analyst helping a finance team understand the output of an "
    "automated reconciliation between a payment platform and a bank. "
    "The reconciliation engine has ALREADY decided which records are discrepancies. "
    "Your job is only to explain the results provided. Rules: use only the numbers and IDs in the "
    "provided JSON; never invent or recalculate figures; say 'possible cause' rather than stating "
    "causes as fact; do not give legal or financial advice; keep the answer concise and practical."
)

# Likely causes and next steps used by the fallback and as hints for the prompt.
TEMPLATES: dict[str, dict[str, list[str]]] = {
    CAT_ROUNDING: {
        "causes": [
            "The platform and the bank round fees, taxes or currency conversions at different steps or with different rules (for example round-half-up vs truncation).",
            "A consistent one-sided difference across many rows points to a systematic rule, not random noise.",
        ],
        "next": [
            "Compare how both systems round (per line item or per total).",
            "Agree a single rounding rule with the bank and book the cumulative amount to a rounding account if it is immaterial.",
        ],
    },
    CAT_DUPLICATE: {
        "causes": [
            "A client or webhook retry created the same transaction twice.",
            "A batch file was loaded twice, or the bank posted the same settlement twice.",
        ],
        "next": [
            "Check idempotency keys and retry logs for the affected IDs.",
            "If the bank paid twice, raise a recovery request with the bank; if the platform double-recorded, reverse the extra entry (do not delete it).",
        ],
    },
    CAT_UNMATCHED_REFUND: {
        "causes": [
            "The original payment belongs to an earlier period or a different system.",
            "The refund was keyed with a wrong or missing original transaction ID.",
        ],
        "next": [
            "Search earlier periods for the original payment.",
            "If no original exists, hold the refund for manual approval and review how refunds are validated at creation.",
        ],
    },
    CAT_REFUND_INCONSISTENT: {
        "causes": ["The refund is larger than the payment, repeated, in another currency or dated before the payment."],
        "next": ["Review the refund against the original payment and confirm the refund amount with the support team."],
    },
    CAT_AMOUNT_MISMATCH: {
        "causes": ["A fee or tax was deducted by one party only, a partial settlement occurred, or an amount was keyed incorrectly."],
        "next": ["Request the bank's settlement detail for the transaction and compare it with the platform record."],
    },
    CAT_MISSING_BANK: {
        "causes": ["The settlement is delayed beyond the normal window, was rejected, or was bundled into another batch."],
        "next": ["Ask the bank for the status of these transactions and check for rejected or held settlements."],
    },
    CAT_MISSING_PLATFORM: {
        "causes": ["The bank settled a payment the platform never recorded, for example a lost webhook or a transaction from another system."],
        "next": ["Trace the bank reference to its origin and record the payment if it is genuine."],
    },
    CAT_STATUS_CONFLICT: {
        "causes": ["The platform marked the payment as failed but the bank settled it, often a timeout where the bank succeeded."],
        "next": ["Confirm the real payment outcome and correct the platform status or reverse the settlement."],
    },
}


@dataclass
class AIReport:
    text: str
    source: str
    model: Optional[str] = None
    error: Optional[str] = None
    unverified_numbers: list[str] = field(default_factory=list)

    @property
    def is_ai(self) -> bool:
        return self.source == SOURCE_AI


# ---------------------------------------------------------------------------
# Context sent to the LLM (aggregated and anonymised)
# ---------------------------------------------------------------------------
def _s(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def build_llm_context(result: ReconciliationResult, max_examples: int = 5) -> dict:
    """Aggregated facts about the run. Contains no customer IDs, bank references or settlement IDs."""
    s = result.summary
    context: dict[str, Any] = {
        "reporting_date": str(s["as_of_date"]),
        "settlement_window_days": f"{result.config.settlement_min_days}-{result.config.settlement_max_days}",
        "rounding_tolerance": _s(result.config.rounding_tolerance),
        "counts": {
            "platform_records": s["total_platform_records"],
            "bank_records": s["total_bank_records"],
            "matched_exactly": s["matched"],
            "rounding_differences": s["rounding_differences"],
            "amount_mismatches": s["amount_mismatches"],
            "pending_settlements": s["pending_settlements"],
            "missing_from_bank": s["missing_from_bank"],
            "missing_from_platform": s["missing_from_platform"],
            "status_conflicts": s["status_conflicts"],
            "platform_duplicate_ids": s["platform_duplicate_ids"],
            "bank_duplicate_ids": s["bank_duplicate_ids"],
            "unmatched_refunds": s["unmatched_refunds"],
            "inconsistent_refunds": s["inconsistent_refunds"],
            "late_settlements": s["late_settlements"],
            "total_discrepancies": s["total_discrepancies"],
        },
        "completion_status": s["completion_status"],
        "bridge_by_currency": [
            {k: _s(v) for k, v in row.items()} for row in result.bridge.to_dict("records")
        ],
        "rounding_by_currency": [
            {"currency": r["currency"], "transactions": r["transactions"], "net_difference": _s(r["net_difference"]),
             "largest_single_difference": _s(r["largest_single_difference"]),
             "cumulative_exceeds_tolerance": bool(r["cumulative_exceeds_tolerance"])}
            for r in result.rounding_by_currency.to_dict("records")
        ],
        "duplicate_examples": [
            {"dataset": d["dataset"], "transaction_id": d["transaction_id"], "occurrences": d["occurrences"],
             "financial_impact": _s(d["financial_impact"]), "impact_reliable": bool(d["impact_reliable"])}
            for d in result.duplicates.head(max_examples).to_dict("records")
        ],
        "top_discrepancies": [
            {"category": d["category"], "severity": d["severity"], "transaction_id": d["transaction_id"],
             "dataset": d["dataset"], "currency": d["currency"], "amount_impact": _s(d["amount_impact"])}
            for d in result.discrepancies.head(max_examples * 2).to_dict("records")
        ],
        "category_counts": result.discrepancies["category"].value_counts().to_dict() if len(result.discrepancies) else {},
    }
    return context


# ---------------------------------------------------------------------------
# Template fallback
# ---------------------------------------------------------------------------
def template_summary(context: dict) -> str:
    c = context["counts"]
    lines = [
        f"### Month-end reconciliation summary (as of {context['reporting_date']})",
        "",
        f"**Status:** {context['completion_status']}.",
        f"The engine compared {c['platform_records']} platform records with {c['bank_records']} bank settlement records. "
        f"{c['matched_exactly']} matched exactly, {c['pending_settlements']} are still within the "
        f"{context['settlement_window_days']} day settlement window (pending, not errors), and "
        f"{c['total_discrepancies']} discrepancies need attention.",
        "",
        "**Gap by currency**",
    ]
    for row in context["bridge_by_currency"]:
        lines.append(
            f"- {row['currency']}: platform {row['platform_total']} vs bank {row['bank_total']} "
            f"(difference {row['difference']}, unexplained {row['unexplained']})."
        )
    lines += ["", "**Findings and next steps**"]
    counts = context["category_counts"]
    if not counts:
        lines.append("- No discrepancies were found.")
    for category, n in counts.items():
        template = TEMPLATES.get(category)
        if not template:
            continue
        lines.append(f"- **{category}** ({n}): possible cause: {template['causes'][0]} Next step: {template['next'][0]}")
    patterns = []
    for r in context["rounding_by_currency"]:
        if r["cumulative_exceeds_tolerance"]:
            patterns.append(
                f"Rounding differences in {r['currency']} are individually tiny (largest {r['largest_single_difference']}) "
                f"but add up to {r['net_difference']} across {r['transactions']} transactions, which suggests a systematic rounding rule."
            )
    if c["bank_duplicate_ids"]:
        patterns.append(f"{c['bank_duplicate_ids']} transaction ID(s) were settled more than once by the bank; this may mean money to recover.")
    if c["late_settlements"]:
        patterns.append(f"{c['late_settlements']} settlement(s) arrived after the normal window.")
    if patterns:
        lines += ["", "**Patterns worth a closer look**"] + [f"- {p}" for p in patterns]
    return "\n".join(lines)


def template_explanation(category: str, details: str = "") -> str:
    template = TEMPLATES.get(category)
    if not template:
        return "No template is available for this category."
    causes = "\n".join(f"- {x}" for x in template["causes"])
    steps = "\n".join(f"- {x}" for x in template["next"])
    head = f"**What the engine found:** {details}\n\n" if details else ""
    return f"{head}**Possible causes**\n{causes}\n\n**Suggested next steps**\n{steps}"


# ---------------------------------------------------------------------------
# Verification of numbers in LLM text
# ---------------------------------------------------------------------------
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _norm(token: str) -> str:
    try:
        return format(Decimal(token).normalize(), "f")
    except InvalidOperation:
        return token


def find_unverified_numbers(text: str, context: dict) -> list[str]:
    """Numbers in ``text`` that do not appear anywhere in the engine context.

    Small whole numbers (0-10) are ignored: they are usually list numbers or words like "1-2 days".
    """
    allowed = {_norm(t) for t in _NUMBER.findall(json.dumps(context, default=str).replace(",", ""))}
    unknown: list[str] = []
    for token in _NUMBER.findall(text.replace(",", "")):
        if "." not in token and int(token) <= 10:
            continue
        if _norm(token) not in allowed and token not in unknown:
            unknown.append(token)
    return unknown


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
def _make_client(api_key: str):
    from groq import Groq  # imported lazily so the app works without the package/key

    return Groq(api_key=api_key, timeout=GROQ_TIMEOUT_SECONDS, max_retries=GROQ_MAX_RETRIES)


def _chat(client, system: str, user: str, max_tokens: int = 1100) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("The model returned an empty response.")
    return text


def _describe_error(exc: Exception, secret: Optional[str] = None) -> str:
    message = " ".join(str(exc).split())[:200]
    if secret:
        message = message.replace(secret, "***")
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _run(context: dict, system: str, user: str, fallback_text: str, api_key: Optional[str], client) -> AIReport:
    key = api_key or get_groq_api_key()
    if client is None and not key:
        return AIReport(fallback_text, SOURCE_TEMPLATE, error="No GROQ_API_KEY configured, so a template explanation is shown.")
    try:
        client = client or _make_client(key)
        text = _chat(client, system, user)
    except Exception as exc:  # noqa: BLE001 - network, auth, rate limit, timeout: always fall back
        return AIReport(fallback_text, SOURCE_TEMPLATE, error=f"AI request failed ({_describe_error(exc, key)}). Showing a template explanation instead.")
    return AIReport(text, SOURCE_AI, model=GROQ_MODEL, unverified_numbers=find_unverified_numbers(text, context))


def generate_summary(result: ReconciliationResult, api_key: Optional[str] = None, client=None) -> AIReport:
    """Month-end summary, unusual patterns and next steps for the whole run."""
    context = build_llm_context(result)
    user = (
        "Write a readable month-end reconciliation summary for a finance manager using the JSON below.\n"
        "Use these sections with short markdown headings: Summary, Unusual patterns, Likely causes, Suggested next steps.\n"
        "Pending settlements are expected timing differences, not errors. Mention only figures that appear in the JSON.\n\n"
        f"JSON:\n{json.dumps(context, default=str)}\n\n"
        f"Hints on common causes by category: {json.dumps(TEMPLATES)}"
    )
    return _run(context, SYSTEM_PROMPT, user, template_summary(context), api_key, client)


def explain_discrepancy(result: ReconciliationResult, discrepancy: dict, api_key: Optional[str] = None, client=None) -> AIReport:
    """Explain one discrepancy row (a dict from the discrepancy table)."""
    category = discrepancy["category"]
    item = {k: _s(v) for k, v in discrepancy.items() if k in ("category", "severity", "transaction_id", "dataset", "currency", "amount_impact", "details")}
    context = {"discrepancy": item, "settlement_window_days": f"{result.config.settlement_min_days}-{result.config.settlement_max_days}",
               "rounding_tolerance": _s(result.config.rounding_tolerance)}
    user = (
        "Explain this single reconciliation discrepancy in under 150 words: what it means, 2-3 possible causes, and what to check next.\n"
        f"JSON:\n{json.dumps(context, default=str)}\n\nCommon causes for this category: {json.dumps(TEMPLATES.get(category, {}))}"
    )
    return _run(context, SYSTEM_PROMPT, user, template_explanation(category, str(discrepancy.get("details", ""))), api_key, client)