"""PayRecon AI - Streamlit dashboard (UI only; all decisions come from reconciliation/)."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from config import get_Gemini_api_key
import pandas as pd
import streamlit as st

from config import (
    APP_NAME, APP_TAGLINE, BANK_CSV, DEFAULT_SEED, MATCH_STATUSES, MATCHED, PLATFORM_CSV, ReconConfig,
    get_Gemini_api_key,
)
from data.generate_data import write_datasets
from reconciliation import ai_explainer
from reconciliation.data_loader import DataLoadError, load_csv
from reconciliation.reconciliation_engine import ReconciliationInputError, run_reconciliation
from utils import charts
from utils.export import discrepancy_csv, full_report_csv, summary_text
from utils.formatting import format_amounts_by_currency, to_display

st.set_page_config(page_title=APP_NAME, page_icon="💳", layout="wide")
st.markdown(
    "<style>.block-container{padding-top:2rem} h1{margin-bottom:0}</style>", unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def show_table(df: pd.DataFrame, key: str, empty_message: str = "Nothing to show here.") -> None:
    """Searchable table. Search matches any column."""
    if df.empty:
        st.info(empty_message)
        return
    query = st.text_input("Search", key=f"search_{key}", placeholder="Type to filter rows")
    shown = df
    if query:
        mask = df.astype(str).apply(lambda col: col.str.contains(query, case=False, regex=False)).any(axis=1)
        shown = df[mask]
    st.caption(f"{len(shown)} of {len(df)} rows")
    st.dataframe(to_display(shown), hide_index=True)


def run_from_inputs(platform_raw: pd.DataFrame, bank_raw: pd.DataFrame, config: ReconConfig) -> None:
    st.session_state.pop("ai_summary", None)
    st.session_state.pop("ai_item", None)
    try:
        st.session_state["result"] = run_reconciliation(platform_raw, bank_raw, config)
        st.session_state["error"] = None
    except ReconciliationInputError as exc:
        st.session_state["result"] = None
        st.session_state["error"] = exc.errors


def load_inputs(use_sample: bool, platform_file, bank_file):
    """Return (platform_raw, bank_raw) or raise DataLoadError with a clear message."""
    if use_sample:
        return load_csv(PLATFORM_CSV, "Sample platform CSV"), load_csv(BANK_CSV, "Sample bank CSV")
    if platform_file is None or bank_file is None:
        raise DataLoadError("Upload both CSV files, or switch to the sample data.")
    return load_csv(platform_file, "Platform CSV"), load_csv(bank_file, "Bank CSV")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Use sample data", "Upload CSV files"], label_visibility="collapsed")
    use_sample = source == "Use sample data"
    platform_file = bank_file = None
    if use_sample:
        seed = st.number_input("Sample seed", min_value=0, value=DEFAULT_SEED, step=1)
        if st.button("Regenerate sample data"):
            write_datasets(int(seed))
            st.session_state["run_requested"] = True
            st.success(f"Sample data regenerated with seed {int(seed)}.")
    else:
        platform_file = st.file_uploader("Platform transactions CSV", type="csv")
        bank_file = st.file_uploader("Bank settlements CSV", type="csv")

    st.header("Settings")
    min_days = st.number_input("Settlement delay: earliest (days)", min_value=0, max_value=30, value=1)
    max_days = st.number_input("Settlement delay: latest (days)", min_value=0, max_value=30, value=2)
    tolerance_text = st.text_input("Rounding tolerance (per transaction)", value="0.05")
    run_clicked = st.button("Run reconciliation", type="primary")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
config = None
try:
    tolerance = Decimal(tolerance_text.strip())
    if not tolerance.is_finite() or tolerance < 0:
        raise InvalidOperation
    if max_days < min_days:
        st.sidebar.error("The latest delay must be at least the earliest delay.")
    else:
        config = ReconConfig(settlement_min_days=int(min_days), settlement_max_days=int(max_days), rounding_tolerance=tolerance)
except InvalidOperation:
    st.sidebar.error("Rounding tolerance must be a number such as 0.05.")

first_load = "result" not in st.session_state and "error" not in st.session_state
if config and (run_clicked or st.session_state.pop("run_requested", False) or (first_load and use_sample)):
    try:
        platform_raw, bank_raw = load_inputs(use_sample, platform_file, bank_file)
        run_from_inputs(platform_raw, bank_raw, config)
    except DataLoadError as exc:
        st.session_state["result"] = None
        st.session_state["error"] = [str(exc)]

result = st.session_state.get("result")
errors = st.session_state.get("error")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(f"💳 {APP_NAME}")
st.caption(APP_TAGLINE)

if errors:
    st.error("The reconciliation could not run:")
    for message in errors:
        st.write(f"- {message}")
    st.stop()
if result is None:
    st.info("Choose your data in the sidebar and select Run reconciliation.")
    st.stop()

s = result.summary
status_text = f"**{s['completion_status']}** (as of {s['as_of_date']})"
if s["total_discrepancies"] == 0 and s["pending_settlements"] == 0:
    st.success(status_text)
elif s["total_discrepancies"] == 0:
    st.info(status_text)
else:
    st.warning(status_text)

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
k = st.columns(6)
k[0].metric("Platform transactions", s["total_platform_records"])
k[1].metric("Bank settlements", s["total_bank_records"])
k[2].metric("Matched", s["matched"])
k[3].metric("Pending settlements", s["pending_settlements"])
k[4].metric("Discrepancies", s["total_discrepancies"])
k[5].metric("Platform minus bank", format_amounts_by_currency(s["total_amount_difference"]))

tabs = st.tabs([
    "Overview", "Matched Transactions", "Rounding Differences", "Duplicate Transactions",
    "Unmatched Refunds", "Pending Settlements", "AI Insights", "Reports and Exports",
])

# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns(2)
    left.plotly_chart(charts.status_breakdown_chart(result.matches), key="c_status")
    right.plotly_chart(charts.discrepancy_distribution_chart(result.discrepancies), key="c_dist")
    left, right = st.columns(2)
    left.plotly_chart(charts.platform_vs_bank_chart(result.bridge), key="c_totals")
    currencies = sorted(result.daily_trend["currency"].unique()) if not result.daily_trend.empty else []
    if currencies:
        currency = right.selectbox("Currency", currencies, key="trend_currency")
        right.plotly_chart(charts.daily_trend_chart(result.daily_trend, currency), key="c_daily")
    if not result.bridge.empty:
        bridge_currency = st.selectbox("Explain the gap for", list(result.bridge["currency"]), key="bridge_currency")
        row = result.bridge[result.bridge["currency"] == bridge_currency].iloc[0].to_dict()
        st.plotly_chart(charts.bridge_chart(row), key="c_bridge")
    st.subheader("All discrepancies")
    show_table(result.discrepancies, "disc", "No discrepancies found.")
    quality = [result.platform_validation, result.bank_validation]
    if any(len(v.invalid_rows) or v.warnings for v in quality):
        with st.expander("Data quality issues (rows excluded from matching)"):
            for v in quality:
                for warning in v.warnings:
                    st.write(f"- {warning}")
                if len(v.invalid_rows):
                    st.write(f"**{v.dataset}**: {len(v.invalid_rows)} problem(s)")
                    st.dataframe(v.invalid_rows, hide_index=True)

# ---------------------------------------------------------------------------
# 2. Matched
# ---------------------------------------------------------------------------
with tabs[1]:
    chosen = st.multiselect("Match status", list(MATCH_STATUSES), default=[MATCHED], key="status_filter")
    show_table(result.with_status(*chosen), "matches", "No transactions with the selected status.")

# ---------------------------------------------------------------------------
# 3. Rounding
# ---------------------------------------------------------------------------
with tabs[2]:
    st.write(
        "Individual differences are each within the tolerance. The cumulative view adds them up, "
        "which is where a systematic rounding rule becomes visible."
    )
    st.plotly_chart(charts.rounding_chart(result.rounding_details), key="c_round")
    st.subheader("Cumulative difference by currency")
    show_table(result.rounding_by_currency, "round_cur", "No rounding differences found.")
    st.subheader("Cumulative difference by settlement week")
    show_table(result.rounding_by_period, "round_week", "No rounding differences found.")
    st.subheader("Individual differences")
    show_table(result.rounding_details, "round_detail", "No rounding differences found.")

# ---------------------------------------------------------------------------
# 4. Duplicates
# ---------------------------------------------------------------------------
with tabs[3]:
    st.write("Duplicates are reported, never deleted. The first record of each ID is used for matching.")
    show_table(result.duplicates, "dups", "No duplicate transaction IDs found.")

# ---------------------------------------------------------------------------
# 5. Refunds
# ---------------------------------------------------------------------------
with tabs[4]:
    if result.refunds.empty:
        st.info("The platform data has no refunds.")
    else:
        statuses = sorted(result.refunds["refund_status"].unique())
        picked = st.multiselect("Refund status", statuses, default=statuses, key="refund_filter")
        show_table(result.refunds[result.refunds["refund_status"].isin(picked)], "refunds")

# ---------------------------------------------------------------------------
# 6. Pending
# ---------------------------------------------------------------------------
with tabs[5]:
    st.write(
        f"Payments newer than {result.config.settlement_max_days} day(s) with no bank record yet. "
        "These are expected timing differences, not errors."
    )
    show_table(result.pending, "pending", "No pending settlements.")

# ---------------------------------------------------------------------------
# 7. AI insights
# ---------------------------------------------------------------------------
def render_report(report: ai_explainer.AIReport) -> None:
    label = "AI-generated explanation" if report.is_ai else "Template-based explanation (no AI used)"
    st.caption(f"{label}" + (f" · model {report.model}" if report.model else ""))
    if report.error:
        st.warning(report.error)
    st.markdown(report.text)
    if report.unverified_numbers:
        st.warning("These figures were not found in the engine output, so check them: " + ", ".join(report.unverified_numbers))


with tabs[6]:
    st.write(
        "The engine finds every discrepancy. The AI only explains results the engine already produced, "
        "and it never sees customer IDs or bank references."
    )
    st.caption(
    "Gemini API key found."
    if get_Gemini_api_key()
    else "No GEMINI_API_KEY set, so template explanations are used."
)
    if st.button("Generate month-end summary"):
        with st.spinner("Writing summary..."):
            st.session_state["ai_summary"] = ai_explainer.generate_summary(result)
    if "ai_summary" in st.session_state:
        render_report(st.session_state["ai_summary"])

    st.divider()
    st.subheader("Explain one discrepancy")
    if result.discrepancies.empty:
        st.info("No discrepancies to explain.")
    else:
        options = {
            f"{r['discrepancy_id']} · {r['category']} · {r['transaction_id']}": r
            for r in result.discrepancies.to_dict("records")
        }
        choice = st.selectbox("Discrepancy", list(options), key="ai_choice")
        if st.button("Explain this discrepancy"):
            with st.spinner("Explaining..."):
                st.session_state["ai_item"] = (choice, ai_explainer.explain_discrepancy(result, options[choice]))
        item = st.session_state.get("ai_item")
        if item and item[0] == choice:
            render_report(item[1])

# ---------------------------------------------------------------------------
# 8. Exports
# ---------------------------------------------------------------------------
with tabs[7]:
    st.write("Download the results of this run.")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Full reconciliation report (CSV)", full_report_csv(result), "reconciliation_report.csv", "text/csv")
    c2.download_button("Discrepancy report (CSV)", discrepancy_csv(result), "discrepancy_report.csv", "text/csv")
    c3.download_button("Summary report (TXT)", summary_text(result), "reconciliation_summary.txt", "text/plain")
    st.text(summary_text(result))
