
# Financial Report Agent

`financial_report_agent_refactor.py` is the main entry point for generating automated financial reports for any stock ticker using Python, yfinance, news sentiment analysis, and Google Generative AI (optional).

## Features
- Fetches financial data and price history
- Computes key metrics (P/E, PEG, Revenue Growth, Debt/Equity)
- Analyzes recent news sentiment
- Generates analyst-style markdown reports ready for export or publishing
- Jupyter notebook for interactive usage
- StateGraph orchestration for advanced workflows (optional)
- Customizable LLM commentary (tone, length)

## Quick Start


### 1. Clone the repository
```bash
git clone https://github.com/Evangeliab/financial-report-agent.git
cd financial-report-agent
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

You can also set API keys in a `.env` file in the project root:
```
GOOGLE_API_KEY=your-google-api-key
NEWSAPI_KEY=your-newsapi-key
GOOGLE_MODEL=gemini-2.5-flash
```

### 5. Run the Jupyter notebook
```bash
jupyter notebook Financial_Report_Usage.ipynb
```
Follow the notebook instructions to generate a report for any stock ticker (e.g., TSLA).


### 6. Run from command line
```bash
# Standard run
python financial_report_agent_refactor.py TSLA

# To use StateGraph orchestration (if available)
python financial_report_agent_refactor.py TSLA --graph
```


## Troubleshooting
- If you see missing package errors, re-run `pip install -r requirements.txt`.
- For API errors, check your API keys and permissions.
- For LLM errors, ensure `langchain-google-genai` is installed and your model name is set via `GOOGLE_MODEL`.
- For Jupyter display issues, update Jupyter and ipywidgets:
  ```bash
  pip install --upgrade jupyter ipywidgets
  ```

## License
MIT
