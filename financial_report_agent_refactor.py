"""
Financial Report Agent: Generates investment reports with financial metrics, news sentiment, LLM-based commentary, and Markdown output.
"""


from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass
import os
import getpass
import io
import base64
import datetime
import math
import textwrap
import logging

import yfinance as yf
import matplotlib.pyplot as plt
from newsapi import NewsApiClient

# Simple dotenv loader (no external dependency) — loads .env from project root if present.
from pathlib import Path

# Require python-dotenv for .env handling (simpler, predictable behavior).
from dotenv import load_dotenv

# Load .env from the project root so subsequent code sees configured keys.
root = Path(__file__).resolve().parent
load_dotenv(dotenv_path=str(root / ".env"), override=False)


def get_model_name() -> str:
    """Return the configured Google model name from environment (or default)."""
    return os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash")


# Logger for warnings/errors (ensure available before we try to instantiate LLM)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Optional imports for LLM and StateGraph
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI



# -----------------------------
# Configuration & API keys
# -----------------------------

# GOOGLE_API_KEY for Gemini (LLM). If not set, prompt securely.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")

try:
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None
except Exception as e:
    logger.error("Failed to instantiate NewsApiClient: %s", repr(e))
    newsapi = None

# Instantiate LLM client if available
llm = None
if ChatGoogleGenerativeAI is not None and os.environ.get("GOOGLE_API_KEY"):
    try:
        model_name = get_model_name()
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, max_retries=2)
        logger.info("Instantiated LLM client with model=%s", model_name)
    except Exception as e:
        attempted = model_name if 'model_name' in locals() else get_model_name()
        logger.error("Failed to instantiate LLM client for model=%s: %s", attempted, repr(e))
        llm = None
else:
    logger.error("langchain_google_genai is not installed or GOOGLE_API_KEY is missing. LLM features will be disabled.")
    llm = None

if llm is None:
    print("WARNING: LLM features are disabled. To enable, set GOOGLE_API_KEY and install langchain_google_genai.")
    print("Some features (news sentiment, analyst commentary) will use fallback logic.")

if newsapi is None:
    logger.error("NewsAPI client is not available. News sentiment features will be disabled. Please install newsapi-python and set NEWSAPI_KEY.")

# -----------------------------
# Types
# -----------------------------
class FinancialReportState(TypedDict):
    symbol: str
    info: Optional[Dict[str, Any]]
    history: Optional[Any]
    financial_metrics: Optional[Dict[str, Any]]
    valuation: Optional[str]
    news_sentiment: Optional[List[Dict[str, Any]]]
    aggregated_sentiment: Optional[Dict[str, Any]]
    narrative: Optional[str]
    markdown_report: Optional[str]


# -----------------------------
# Utilities / Data fetching
# -----------------------------

def safe_get_ticker(symbol: str):
    """Return yf.Ticker and fetched info/history with safe fallbacks."""
    ticker = yf.Ticker(symbol)
    # fast_info is more reliable for price-related fields
    fast = getattr(ticker, "fast_info", {}) or {}
    try:
        hist = ticker.history(period="1y", interval="1d", actions=False)
    except Exception:
        hist = None
    # info may be partial; use .info but handle failures
    try:
        info = ticker.info
    except Exception:
        info = {}
    return ticker, info, fast, hist


# -----------------------------
# Valuation Agent
# -----------------------------

def valuation_agent(state: FinancialReportState) -> Dict[str, Any]:
    """
    Compute key financial metrics and generate a rule-based investment recommendation for a given stock symbol.

    Extracts metrics such as trailing P/E, PEG ratio, revenue growth, debt/equity, and market cap from Yahoo Finance data.
    Applies simple rules to recommend 'Buy', 'Hold', or 'Sell' based on these metrics.

    Args:
        state (FinancialReportState): The current report state, must include 'symbol'.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - info: Raw company info from yfinance
            - history: Price history DataFrame
            - financial_metrics: Dict of computed metrics
            - valuation: String recommendation ('Buy', 'Hold', 'Sell')
    """
    symbol = state["symbol"]
    try:
        ticker, info, fast, hist = safe_get_ticker(symbol)
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return {
            "info": {},
            "history": None,
            "financial_metrics": {"error": "Failed to fetch data. API rate limit or connection issue."},
            "valuation": "N/A",
        }

    # Check for missing or empty data (rate limit, delisted, etc.)
    if not info or (hist is not None and hasattr(hist, "empty") and hist.empty):
        logger.error(f"No data returned for {symbol}. Possible rate limit or delisted ticker.")
        return {
            "info": info,
            "history": hist,
            "financial_metrics": {"error": "No data returned. Possible rate limit or delisted ticker."},
            "valuation": "N/A",
        }

    trailing_pe = info.get("trailingPE") or fast.get("trailing_pe")
    revenue_growth = info.get("revenueGrowth")
    revenue_growth_pct = float(revenue_growth) * 100 if revenue_growth else None
    debt_to_equity = info.get("debtToEquity")
    market_cap = info.get("marketCap") or fast.get("market_cap")
    earnings_growth = info.get("earningsQuarterlyGrowth") or info.get("earningsGrowth")

    peg = None
    if trailing_pe and earnings_growth:
        try:
            if isinstance(earnings_growth, (int, float)) and earnings_growth != 0:
                peg = trailing_pe / (earnings_growth * 100)
            else:
                peg = trailing_pe / float(earnings_growth)
        except Exception:
            peg = None

    metrics = {
        "P/E (trailing)": trailing_pe or "N/A",
        "PEG": round(peg, 2) if isinstance(peg, (int, float)) and not math.isnan(peg) else "N/A",
        "Revenue Growth (%)": f"{revenue_growth_pct:.2f}%" if revenue_growth_pct is not None else "N/A",
        "Debt/Equity": round(debt_to_equity, 2) if isinstance(debt_to_equity, (int, float)) else "N/A",
        "Market Cap (USD)": f"${market_cap:,}" if market_cap else "N/A",
    }

    recommendation = "Hold"
    if trailing_pe and trailing_pe < 15 and revenue_growth_pct and revenue_growth_pct > 10 and (not isinstance(debt_to_equity, (int, float)) or debt_to_equity < 1.5):
        recommendation = "Buy"
    elif isinstance(peg, (int, float)) and peg > 0 and peg < 1.5:
        recommendation = "Buy"
    elif trailing_pe and trailing_pe > 40 and revenue_growth_pct and revenue_growth_pct < 0:
        recommendation = "Sell"

    return {
        "info": info,
        "history": hist,
        "financial_metrics": metrics,
        "valuation": recommendation,
    }


# -----------------------------
# News Agent (LLM-based sentiment scoring)
# -----------------------------

def _llm_score_article(article_title: str, article_description: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of a financial news article using an LLM.

    Args:
        article_title (str): The headline of the news article.
        article_description (str): The description or summary of the article.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - label: Sentiment label ('Positive', 'Neutral', 'Negative')
            - score: Integer sentiment score in [-2, 2]
            - rationale: Short explanation of the sentiment decision
    """
    # Only use LLM for sentiment scoring. If unavailable, return Neutral/0 with rationale.
    if llm is not None:
        prompt = textwrap.dedent(f"""
        You are a concise financial news grader.
        Given a news headline and short description, return a JSON object with three fields:
        - label: one of Positive, Neutral, Negative
        - score: integer in [-2, -1, 0, 1, 2] where 2 is very positive, -2 very negative
        - rationale: 1-2 sentence explanation in plain English.

        Headline: {article_title}
        Description: {article_description or ""}

        Return ONLY JSON.
        """)
        try:
            response = llm.invoke(prompt)
            text = getattr(response, "content", str(response))
            import json
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_text = text[start:end+1]
                parsed = json.loads(json_text)
                return {
                    "label": parsed.get("label", "Neutral"),
                    "score": parsed.get("score", 0),
                    "rationale": parsed.get("rationale", "")
                }
        except Exception as e:
            logger.warning(f"LLM sentiment scoring failed: {e}")
            # fallback below

    # If LLM is unavailable or fails, return Neutral sentiment
    return {
        "label": "Neutral",
        "score": 0,
        "rationale": "No LLM available for sentiment analysis. Defaulting to Neutral."
    }


def news_agent(state: FinancialReportState, max_articles: int = 6) -> Dict[str, Any]:
    """
    Fetch and analyze recent news articles for a given stock symbol.

    Each article is scored for sentiment using the LLM (if available).
    Aggregates sentiment statistics across all articles.

    Args:
        state (FinancialReportState): The current report state, must include 'symbol'.
        max_articles (int): Maximum number of news articles to fetch and score.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - news_sentiment: List of per-article sentiment dicts
            - aggregated_sentiment: Dict with counts, average_score, n_articles
    """
    symbol = state["symbol"]
    if newsapi is None:
        logger.warning("NewsAPI client unavailable. Returning empty news sentiment.")
        return {"news_sentiment": [], "aggregated_sentiment": {"counts": {}, "average_score": 0, "n_articles": 0}}

    try:
        raw = newsapi.get_everything(q=symbol, language="en", sort_by="relevancy", page_size=max_articles)
        articles = raw.get("articles", [])
    except Exception as e:
        logger.warning(f"NewsAPI fetch failed: {e}")
        articles = []

    scored = []
    for a in articles:
        title = a.get("title", "")
        desc = a.get("description") or a.get("content") or ""
        result = _llm_score_article(title, desc)
        scored.append({
            "title": title,
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "label": result["label"],
            "score": result["score"],
            "rationale": result["rationale"],
        })

    # Aggregate sentiment
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    score_sum = 0
    for s in scored:
        if s["label"] in counts:
            counts[s["label"]] += 1
        try:
            score_sum += int(s["score"])
        except Exception:
            pass

    aggregated = {
        "counts": counts,
        "average_score": score_sum / len(scored) if scored else 0,
        "n_articles": len(scored),
    }

    return {
        "news_sentiment": scored,
        "aggregated_sentiment": aggregated,
    }


# -----------------------------
# Narrative Agent
# -----------------------------

def narrative_agent(state: FinancialReportState, tone: str = "formal", length: str = "medium") -> Dict[str, Any]:
    """
    Generate a short analyst-style investment commentary using an LLM.

    Composes a multi-paragraph markdown report summarizing financial metrics, news sentiment, risks, and next steps.
    If LLM is unavailable, returns a templated summary instead.

    Args:
        state (FinancialReportState): The current report state, must include financial metrics and news sentiment.

    Returns:
        Dict[str, Any]: Dictionary with key:
            - narrative: Markdown-formatted analyst commentary string
    """
    today = datetime.date.today().strftime("%B %d, %Y")
    metrics = state.get("financial_metrics") or {}
    agg = state.get("aggregated_sentiment") or {}

    
    # Enhanced prompt for clarity, actionable insights, and audience targeting
    prompt = textwrap.dedent(f"""
    You are an experienced equity analyst writing for professional investors and portfolio managers.
    Today's date: {today}

    Company: {state['symbol']}

    Your task is to draft a concise (3-5 paragraph) investment commentary suitable for a research note. Structure your response as follows:
    1. **Headline:** One sentence summarizing the investment recommendation (Buy/Hold/Sell) and rationale.
    2. **Financial Overview:** Summarize recent financial metrics, explicitly referencing P/E, Revenue Growth, Debt/Equity, PEG (if available), and any notable trends or anomalies.
    3. **News Synthesis:** Integrate the aggregated news sentiment and highlight 1-2 notable articles, explaining their relevance and impact on the investment thesis.
    4. **Risks:** List and briefly explain 3 key risks that could affect the investment outlook.
    5. **Next Steps:** Suggest what investors should monitor next (e.g., upcoming earnings, management guidance, macroeconomic indicators).

    Use the following structured data in your analysis:
    Financial Metrics: {metrics}
    Aggregated News Sentiment: {agg}

    Output requirements:
    - Write in {tone} style.
    - Target length: {length}.
    - Format as plain markdown text (no JSON, no code blocks).
    - Make the analysis actionable and relevant for professional investors.
    """)



    # # Build a careful prompt that provides the model with structure and style control
    # prompt = textwrap.dedent(f"""
    # You are an experienced equity analyst.
    # Today's date: {today}

    # Company: {state['symbol']}

    # Provide a short (3-5 paragraph) investment commentary suitable for a research note. The commentary should include:
    # 1) A concise headline (one line) summarizing the recommendation.
    # 2) A short overview of recent financial metrics (explicitly mention P/E, Revenue Growth, Debt/Equity, PEG if available).
    # 3) A brief synthesis of recent news (synthesize the aggregated sentiment and 1-2 notable articles with rationale).
    # 4) A Risks section listing 3 key risks.
    # 5) A "Next Steps" section suggesting what to monitor next (e.g., earnings, guidance, macro indicators).

    # Use the following structured data in your analysis:
    # Financial Metrics: {metrics}
    # Aggregated News Sentiment: {agg}

    # Output style: {tone}
    # Desired length: {length}

    # Return plain markdown text; do not include any JSON. Keep the tone as specified.
    # """)


    # # LLM is required here (we raised earlier if missing). Invoke and propagate any errors
    # response = llm.invoke(prompt)
    # narrative_text = getattr(response, "content", str(response))
    # return {"narrative": narrative_text}


# -----------------------------
# Plotting & Report Agent
# -----------------------------

def plot_stock_price_with_indicators(history, symbol: str):
    """Create a PNG in memory of the price chart with moving averages and volume.

    Returns a Markdown image string with a data URI.
    """
    if history is None or history.empty:
        return """(No historical data available)"""

    # Compute moving averages
    history = history.copy()
    history['MA20'] = history['Close'].rolling(window=20).mean()
    history['MA50'] = history['Close'].rolling(window=50).mean()
    history['MA200'] = history['Close'].rolling(window=200).mean()

    import matplotlib.dates as mdates

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Price and moving averages with colors and styles
    ax[0].plot(history.index, history['Close'], label='Close')
    ax[0].plot(history.index, history['MA20'], label='MA20', )
    ax[0].plot(history.index, history['MA50'], label='MA50')
    ax[0].plot(history.index, history['MA200'], label='MA200')
    ax[0].set_title(f"{symbol} - Close Price with Moving Averages (1y)")
    ax[0].set_ylabel("Price")
    ax[0].legend()
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)


    # Volume on the lower axis with color and transparency
    ax[1].bar(history.index, history['Volume'],  alpha=0.5)
    ax[1].set_ylabel("Volume")
    ax[1].set_xlabel("Date")
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Format volume axis with commas
    import matplotlib.ticker as mticker
    #ax[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.0f}M"))

    # Improve date formatting
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return f"![Stock Price Chart](data:image/png;base64,{image_base64})"


def report_agent(state: FinancialReportState) -> Dict[str, Any]:
    """
    Assemble a Markdown-formatted investment report from the current state.

    Includes price chart, key metrics, analyst recommendation, news sentiment, narrative, risks, next steps, and disclaimer.

    Args:
        state (FinancialReportState): The current report state containing all relevant data.

    Returns:
        Dict[str, Any]: Dictionary with key 'markdown_report' containing the full Markdown report.
    """
    import logging
    today = datetime.date.today().strftime("%B %d, %Y")
    md = [f"# {state['symbol']} Investment Commentary - {today}\n"]

    # Price chart
    hist = state.get('history')
    if hist is None:
        logging.warning(f"No historical price data for {state['symbol']}")
    md.append("## 📈 Price Chart")
    md.append(plot_stock_price_with_indicators(hist, state['symbol']))

    # Key metrics
    metrics = state.get('financial_metrics')
    if not metrics:
        logging.warning(f"No financial metrics for {state['symbol']}")
    md.append("## 📊 Key Financial Metrics")
    for k, v in (metrics or {}).items():
        md.append(f"- **{k}**: {v}")

    # Valuation badge with emoji
    val = state.get('valuation', 'N/A')
    badge = {
        "Buy": "🟢 Buy",
        "Hold": "🟡 Hold",
        "Sell": "🔴 Sell"
    }.get(val, f"⚪ {val}")
    md.append(f"## 🏅 Analyst Recommendation\n**{badge}**")

    # News section
    news_sentiment = state.get('news_sentiment')
    if not news_sentiment:
        logging.warning(f"No news sentiment for {state['symbol']}")
    md.append("## 📰 News Sentiment Summary")
    agg = state.get('aggregated_sentiment') or {}
    counts = agg.get('counts', {})
    md.append(f"- Positive: {counts.get('Positive', 0)}  |  Neutral: {counts.get('Neutral', 0)}  |  Negative: {counts.get('Negative', 0)}")
    for a in (news_sentiment or []):
        md.append(f"- [{a.get('title')}]({a.get('url')}) — {a.get('label')} ({a.get('score')}) — {a.get('rationale')}")

    # Narrative
    narrative = state.get('narrative')
    if not narrative:
        logging.warning(f"No narrative generated for {state['symbol']}")
    md.append("## 🤖 Analyst Commentary")
    md.append(narrative or "(No narrative generated)")

    # Risks section (if present)
    risks = state.get('risks')
    if risks:
        md.append("## ⚠️ Key Risks")
        for r in risks:
            md.append(f"- {r}")

    # Next Steps section (if present)
    next_steps = state.get('next_steps')
    if next_steps:
        md.append("## ⏭️ Next Steps")
        for step in next_steps:
            md.append(f"- {step}")

    # Disclaimer
    md.append("## ⚠️ Disclaimer ⚠️")
    md.append("The generated report and analysis do not constitute financial advice. Please consult a qualified financial advisor before making investment decisions.")

    report_md = "\n\n".join(md)
    return {"markdown_report": report_md}


# -----------------------------
# Orchestration (sequential runner)
# -----------------------------

def run_financial_report(symbol: str, use_graph: bool = False, max_articles: int = 6) -> FinancialReportState:
    """
    Orchestrate the financial report generation by running all agents in sequence or via StateGraph.

    Handles error logging for each agent. Optionally uses StateGraph for advanced orchestration.

    Args:
        symbol (str): Ticker symbol for the report.
        use_graph (bool): Whether to use StateGraph orchestration.
        max_articles (int): Maximum number of news articles to fetch.

    Returns:
        FinancialReportState: Final state containing all report data and Markdown output.
    """
    state: FinancialReportState = {
        "symbol": symbol,
        "info": None,
        "history": None,
        "financial_metrics": None,
        "valuation": None,
        "news_sentiment": None,
        "aggregated_sentiment": None,
        "narrative": None,
        "markdown_report": None,
    }

    import logging
    # Agents executed sequentially with error handling
    agents = [
        (valuation_agent, {}),
        (news_agent, {"max_articles": max_articles}),
        (narrative_agent, {"tone": "formal", "length": "medium"}),
        (report_agent, {}),
    ]
    for agent, kwargs in agents:
        try:
            state.update(agent(state, **kwargs))
        except Exception as e:
            logging.error(f"{agent.__name__} failed for {symbol}: {e}")

    # Optionally build/visualize graph if StateGraph is available
    if StateGraph is not None and use_graph:
        try:
            g = StateGraph(FinancialReportState)
            g.add_node("ValuationAgent", valuation_agent)
            g.add_node("NewsAgent", news_agent)
            g.add_node("NarrativeAgent", narrative_agent)
            g.add_node("ReportAgent", report_agent)
            g.set_entry_point("ValuationAgent")
            g.add_edge("ValuationAgent", "NewsAgent")
            g.add_edge("NewsAgent", "NarrativeAgent")
            g.add_edge("NarrativeAgent", "ReportAgent")
            g.set_finish_point("ReportAgent")
            compiled = g.compile()
            # Actually run the graph and merge its output into state
            graph_result = compiled.run(state)
            if isinstance(graph_result, dict):
                state.update(graph_result)
        except Exception as e:
            import logging
            logging.error(f"StateGraph execution failed for {symbol}: {e}")

    return state


# -----------------------------
# CLI entry point for running the Financial Report Agent from the command line
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Financial Report Agent for a given ticker symbol")
    parser.add_argument("symbol", type=str, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--graph", action='store_true', help="Build the StateGraph for visualization (if available)")
    args = parser.parse_args()

    result = run_financial_report(args.symbol, use_graph=args.graph)

    # Print top-level summary
    print("== Financial Metrics ==")
    for k, v in (result.get('financial_metrics') or {}).items():
        print(f"- {k}: {v}")
    print("\n== Aggregated News Sentiment ==")
    print((result.get('aggregated_sentiment') or {}))
    print("\n== Markdown Report Preview ==\n")
    print(result.get('markdown_report')[:2000])



