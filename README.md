# PayRecon AI

Payment reconciliation for a payment platform and its bank. The platform records a transaction instantly; the bank settles 1-2 days later. At month end the two datasets should agree. PayRecon AI compares them, finds the differences, and explains them.

**A deterministic Pandas engine makes every financial decision. The LLM (Groq) only explains results the engine has already produced.** Without an API key the app still works and uses labelled template explanations.

## What it detects

| Finding | Meaning |
|---|---|
| Rounding Difference | Per-transaction gap within the tolerance. Also shown cumulatively by currency and settlement week, so many tiny gaps that add up become visible. |
| Duplicate Entry | The same transaction ID more than once, reported separately for the platform and the bank. Nothing is deleted. |
| Unmatched Refund | A refund whose original payment is missing, blank, self-referencing, a refund, or a failed payment. Refunds that exceed or are inconsistent with the original are flagged separately. |
| Pending Settlement | Not yet in the bank file but still within the settlement window. **Not an error.** |
| Also found | Amount Mismatch, Missing from Bank, Missing from Platform, Status Conflict. |

An amount bridge explains the platform-minus-bank gap per currency. Its "unexplained" line is a self-check and should always be zero.

## Project layout

```
app.py                         Streamlit UI (no business logic)
config.py                      Constants, ReconConfig, Groq key lookup
reconciliation/
  data_loader.py               CSV -> raw all-string DataFrames
  validator.py                 Validation, Decimal/date parsing, invalid-row report
  matcher.py                   Platform <-> bank matching and classification
  discrepancy_detector.py      Duplicates, rounding, refunds, discrepancy table
  reconciliation_engine.py     run_reconciliation(): single entry point, bridge, summary
  ai_explainer.py              Groq explanations + template fallback
utils/                         charts.py (Plotly), export.py, formatting.py
data/                          generate_data.py + the two sample CSVs
tests/                         79 pytest tests
```

## Run locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens with the sample data already reconciled. Use the sidebar to upload your own files, change the settlement window or rounding tolerance, or regenerate the sample data.

### Input format

`platform_transactions.csv`: `transaction_id, customer_id, transaction_date, transaction_type (PAYMENT|REFUND), amount, currency, status (SUCCESS|FAILED), original_transaction_id`

`bank_settlements.csv`: `settlement_id, transaction_id, settlement_date, settlement_amount, currency, bank_reference`

Refund amounts on the platform are positive with type `REFUND`; the bank records them as negative amounts. Supported currencies: INR, USD, EUR, GBP.

## Groq API key

The key is read on the server only, from the environment or Streamlit secrets, and is never shown in the UI.

- Local: copy `.env.example` to `.env` and set `GROQ_API_KEY`. `.env` is git-ignored.
- Streamlit Cloud: see "Deploy" below.

Get a key at https://console.groq.com. Without a key, the AI Insights tab shows template explanations labelled "Template-based". Failed or timed-out API calls (20 s timeout, 1 retry) fall back the same way. Only aggregated, anonymised figures are sent to the model: no customer IDs, bank references or settlement IDs. Numbers in AI text that do not appear in the engine output are flagged.

## Run the tests

```bash
python -m pytest -q
```

79 tests, all passing at the time of writing. They assert actual outputs, not just that code runs. They cover the four required scenarios (normal match, 100.00 vs 99.99 rounding, duplicate entry, unmatched refund), pending settlement inside and outside the window, missing IDs, invalid columns, empty files, duplicates in both datasets, cumulative rounding, the AI fallback and failure paths, and exports. One test checks that the engine finds exactly the problems injected by the data generator.

## Matching assumptions

1. Records are matched on `transaction_id`. Currency must also agree, otherwise the pair is an Amount Mismatch.
2. Money is `decimal.Decimal` throughout. Floats are used only for charts and on-screen tables.
3. Refunds are compared as negative amounts, so a refund settled as a negative bank amount matches.
4. If an ID appears several times in one dataset, the first occurrence is matched and the extra copies are reported as duplicates.
5. The reporting date is the latest date found in either file (a different date can be set in `ReconConfig.as_of_date`).
6. A SUCCESS payment with no bank record is Pending when its age (reporting date minus transaction date) is at most the settlement window maximum, otherwise Missing from Bank.
7. A gap up to and including the rounding tolerance (default 0.05) is a Rounding Difference. A larger gap is an Amount Mismatch.
8. A settlement outside the 1-2 day window is still matched, but flagged as a late settlement.
9. FAILED platform payments expect no settlement (Not Settleable). If the bank settled one, it is a Status Conflict.
10. Invalid rows (bad amount, date, currency, or missing ID) are excluded from matching but listed under data quality issues; they are not silently dropped. A file with missing columns, no rows, or no valid rows stops the run with a clear message.
11. Refunds are checked against payments in the same platform file. A refund for a payment in an earlier period appears as Unmatched until the earlier period's data is included.

## Deploy to GitHub and Streamlit Community Cloud

```bash
git init
git add .
git commit -m "PayRecon AI"
git branch -M main
git remote add origin https://github.com/<your-user>/PayRecon-AI.git
git push -u origin main
```

Then:

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Select **Create app**, choose the repository, branch `main`, and main file `app.py`.
3. Open **Advanced settings > Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
4. Deploy. Secrets can be edited later under the app's **Settings > Secrets**. Never commit `.streamlit/secrets.toml`.

The sample CSVs are committed, so the app works on first load. "Regenerate sample data" rewrites those files on the server; on Streamlit Cloud they reset when the app restarts.

## Production risks and how to reduce them

**Incorrect or delayed bank settlement data.** A late or partial bank file would make valid payments look missing. Reduce it by keeping the settlement window configurable, flagging late settlements instead of treating them as errors, checking the bank file's completeness (record counts or control totals) before reconciling, and re-running when late data arrives.

**Duplicate processing or incorrect matching.** Matching on ID alone can pair the wrong records, and "fixing" duplicates by deleting them destroys evidence. This engine never deletes data, reports duplicates separately per dataset, matches on ID and currency, and has the zero-unexplained bridge as a self-check. In production, add idempotency keys on the platform side, match on additional fields (reference, date), and require human approval before adjustments are posted.

**Financial data privacy and security.** Transaction data is sensitive and an LLM is an external service. Here, only aggregated figures go to the model, API keys live in environment variables or secrets, and the model cannot change results. In production, also use access control and audit logs, encrypt data at rest and in transit, use a provider under a data-processing agreement (or a self-hosted model), and keep the CSVs out of source control.

## Limitations

- Matching is by transaction ID; there is no fuzzy matching on amount and date.
- One month of data is expected per run.
- Platform status is limited to SUCCESS and FAILED.
- The app has been tested with Streamlit's test harness and the unit tests; deployment to Streamlit Community Cloud and a live Groq call must be verified by you with your own account and key.

## Create a ZIP archive

```bash
cd ..
zip -r PayRecon-AI.zip PayRecon-AI -x "*/__pycache__/*" "*/.pytest_cache/*" "*/.venv/*" "*/.env"
```

(On Windows, right-click the project folder > Send to > Compressed folder, after deleting `.venv` and `.env`.)

## Demo video checklist

- [ ] Run `streamlit run app.py` and show the dashboard loading with sample data.
- [ ] Explain the problem in one sentence (instant platform record, bank settles 1-2 days later).
- [ ] Point out the KPI cards and the status banner.
- [ ] Walk through the Overview charts, then the gap waterfall (unexplained = 0).
- [ ] Rounding tab: individual vs cumulative differences.
- [ ] Duplicates tab: platform vs bank, financial impact.
- [ ] Unmatched Refunds tab: unmatched vs valid refunds.
- [ ] Pending Settlements tab: expected timing, not errors. Change the settlement window in the sidebar and re-run to show it update.
- [ ] Upload a broken CSV (missing column) to show the clear error message.
- [ ] AI Insights: generate the summary and explain one discrepancy; point out the AI / template label.
- [ ] Reports tab: download the three reports.
- [ ] Terminal: run `python -m pytest -q` and show all tests passing.
- [ ] Mention the design rule: the engine decides, the LLM only explains.
