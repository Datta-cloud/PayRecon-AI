# PayRecon AI 💳
### AI-Powered Payment Reconciliation & Discrepancy Detection

PayRecon AI is an intelligent payment reconciliation application that compares payment platform transactions with bank settlement records to identify discrepancies, detect unusual patterns, and generate actionable explanations using **Google Gemini AI**.

It helps finance teams identify rounding differences, duplicate transactions, unmatched refunds, missing payments, and pending settlements through an interactive dashboard.

---

## 🚀 Key Features

- **Automated Reconciliation:** Compare payment platform records against bank settlement data.
- **Rounding Difference Detection:** Identify small transaction-level differences and cumulative rounding variations.
- **Duplicate Transaction Detection:** Identify potential duplicate entries in platform and bank records.
- **Unmatched Refund Detection:** Find refunds without valid original payment references.
- **Pending Settlement Detection:** Handle normal bank settlement delays of 1–2 days.
- **Missing Transaction Detection:** Identify transactions appearing in only one dataset.
- **AI-Powered Insights:** Use Google Gemini to explain detected discrepancies and generate month-end summaries.
- **Interactive Dashboard:** Explore results through dedicated tabs and summary metrics.
- **Reports & Exports:** Review and export reconciliation findings.
- **Fallback Explanations:** Continue using template-based explanations if Gemini is unavailable.
- **Configurable Settings:** Adjust settlement delays, rounding tolerance, and sample data generation.

---

## 🧠 How It Works

PayRecon AI follows a deterministic-first architecture.

1. Upload payment platform and bank settlement CSV files, or use sample data.
2. Validate and preprocess the datasets.
3. Run the reconciliation engine to match transactions.
4. Detect discrepancies such as duplicates, rounding differences, unmatched refunds, and missing transactions.
5. Send anonymised reconciliation summaries to Google Gemini for explanations.
6. Display results and actionable recommendations in the dashboard.

**Important:** The reconciliation engine performs all financial calculations and discrepancy detection. Gemini AI only explains the results; it does not decide whether transactions match.

---

## 🏗️ System Architecture

```text
       Platform Transactions CSV
                  |
                  |
       Bank Settlements CSV
                  |
                  v
        Data Validation
                  |
                  v
      Reconciliation Engine
                  |
          +-------+-------+
          |       |       |
          v       v       v
       Rounding Duplicates Refunds
          |       |       |
          +-------+-------+
                  |
                  v
        Reconciliation Results
                  |
                  v
           Google Gemini AI
                  |
          AI Explanations
                  |
                  v
        Streamlit Dashboard
                  |
          Reports & Exports
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend / Dashboard | Streamlit |
| Data Processing | Pandas |
| Financial Calculations | Python Decimal |
| AI Integration | Google Gemini API |
| AI SDK | Google GenAI (`google-genai`) |
| Visualization | Plotly |
| Testing | Pytest |
| Environment Configuration | python-dotenv |
| Deployment | Streamlit Community Cloud |
| Version Control | Git & GitHub |


---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Datta-cloud/PayRecon-AI.git
```

### 2. Navigate to the project directory

```bash
cd PayRecon-AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If the Gemini SDK is not already included in your requirements:

```bash
pip install google-genai python-dotenv
```

### 5. Configure Gemini API

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Replace the placeholder with your actual API key.

**Never commit your `.env` file or expose your API key publicly.**

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🤖 Google Gemini AI Integration

PayRecon AI uses Google's Gemini API to generate natural-language explanations from reconciliation results.

Gemini helps with:

- Explaining why discrepancies may have occurred.
- Identifying unusual reconciliation patterns.
- Generating month-end reconciliation summaries.
- Suggesting practical investigation steps.
- Helping finance teams understand reconciliation outcomes.

### AI workflow

```text
Reconciliation Engine
        |
        v
Anonymised Summary
        |
        v
Google Gemini API
        |
        v
AI-Generated Explanation
        |
        v
Dashboard
```

The application also supports template-based explanations when:
- No Gemini API key is configured.
- The API request fails.
- The service is temporarily unavailable.

The deterministic reconciliation engine continues to work independently of the AI service.

---

## 📊 Dashboard Modules

| Module | Description |
|---|---|
| Overview | Overall reconciliation statistics |
| Matched Transactions | Successfully matched transactions |
| Rounding Differences | Transaction-level and cumulative rounding differences |
| Duplicate Transactions | Potential duplicate transaction records |
| Unmatched Refunds | Refunds without valid original payments |
| Pending Settlements | Transactions within the expected settlement window |
| AI Insights | Gemini-powered explanations and summaries |
| Reports and Exports | Reconciliation reports and downloadable results |

---

## 🧪 Testing

Run the test suite using:

```bash
pytest -q
```

Important test scenarios include:

1. Exact transaction matching.
2. Rounding difference detection.
3. Duplicate transaction detection.
4. Unmatched refund detection.
5. Pending settlements within the 1–2 day settlement window.
6. Missing transactions.
7. Gemini API failure and template fallback.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push the project to GitHub.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new application.
4. Select the `Datta-cloud/PayRecon-AI` repository.
5. Select the `main` branch.
6. Set the main file path to `app.py`.
7. Open application settings and navigate to **Secrets**.
8. Add the following secrets:

```toml
GEMINI_API_KEY = "your_actual_gemini_api_key"
GEMINI_MODEL = "gemini-2.5-flash"
```

9. Save and deploy the application.

---

## 🔐 Security & Production Considerations

- API keys are stored in environment variables or deployment secrets.
- Sensitive customer identifiers and bank references should not be sent to Gemini.
- Financial calculations are handled by deterministic Python logic.
- AI-generated explanations are advisory and should be reviewed before financial action.
- Duplicate detection should distinguish confirmed duplicates from probable duplicates.
- Settlement timing differences should not automatically be treated as financial errors.
- Production deployments should include access controls, audit logs, monitoring, and secure data retention policies.

---

## 🔮 Future Enhancements

- Automated bank statement integrations.
- Real-time reconciliation monitoring.
- Advanced duplicate detection.
- Multi-bank and multi-payment gateway support.
- Historical reconciliation analytics.
- Role-based access control.
- Improved audit trails and exception workflows.

---

## 👨‍💻 Author

**Datta Hadke**

B.Tech – Artificial Intelligence & Data Science

Vishwakarma Institute of Information Technology, Pune

GitHub: [Datta-cloud](https://github.com/Datta-cloud)

---

## 📜 License

This project is developed for educational and demonstration purposes. Add an appropriate open-source license before distributing it for reuse.
