# Financial Report Agent

This project generates automated financial reports for any stock ticker using Python, yfinance, news sentiment analysis, and Google Generative AI (optional).

## Features
- Fetches financial data and price history
- Computes key metrics (P/E, PEG, Revenue Growth, Debt/Equity)
- Analyzes recent news sentiment
- Generates analyst-style markdown reports
- Jupyter notebook for interactive usage

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Evangeliab/financial-report-agent.git
cd financial_report
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
- **Google Generative AI:** Set `GOOGLE_API_KEY` in your environment for LLM features.
- **NewsAPI:** Set `NEWSAPI_KEY` in your environment for news sentiment analysis.

Example:
```bash
export GOOGLE_API_KEY=your-google-api-key
export NEWSAPI_KEY=your-newsapi-key
```

### 5. Run the Jupyter notebook
```bash
jupyter notebook Financial_Report_Usage.ipynb
```
Follow the notebook instructions to generate a report for any stock ticker (e.g., TSLA).

### 6. Run from command line
```bash
python financial_report_agent_refactor.py TSLA
```

## Troubleshooting
- If you see missing package errors, re-run `pip install -r requirements.txt`.
- For API errors, check your API keys and permissions.
- For Jupyter display issues, update Jupyter and ipywidgets:
  ```bash
  pip install --upgrade jupyter ipywidgets
  ```

## License
MIT
