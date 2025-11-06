# Refactored Financial Report Agent
# Improvements implemented:
# - Robust data fetching (yfinance fast_info + history)
# - Expanded valuation logic (P/E, PEG if available, Debt/Equity, Revenue Growth)
# - LLM-based news sentiment scoring (per-article, then aggregated)
# - Moving averages and volume on the price chart
# - Structured narrative generation via LLM (explicit system + user prompts)
# - Safer API key handling and error handling
# - Modular, testable agents and an orchestration function
# - Produces a Markdown report ready for Medium (or export)

from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass
import os
import getpass
import io
import base64
import datetime
import math
import textwrap

import yfinance as yf
import matplotlib.pyplot as plt
from newsapi import NewsApiClient

# Optional imports for LLM and StateGraph

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

try:
    from langgraph.graph import StateGraph
except Exception:
    StateGraph = None


# -----------------------------
# Configuration & API keys
# -----------------------------

# GOOGLE_API_KEY for Gemini (LLM). If not set, prompt securely.
if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key (or set GOOGLE_API_KEY env): ")
    except Exception:
        # In non-interactive environments, continue without LLM
        pass

# NEWSAPI key. Prefer environment var for security; otherwise prompt.
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    try:
        NEWSAPI_KEY = getpass.getpass("Enter your NewsAPI key (or set NEWSAPI_KEY env): ")
    except Exception:
        NEWSAPI_KEY = None

newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None

# Instantiate LLM client if available
llm = None
if ChatGoogleGenerativeAI is not None and os.environ.get("GOOGLE_API_KEY"):
    # Allow model selection via environment variable so users can pick a supported model
    # Default to a more recent/available Gemini variant that is commonly supported
    MODEL_NAME = os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash")
    try:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0, max_retries=2)
    except Exception:
        # If instantiation fails (e.g., unsupported model name), disable LLM usage
        llm = None


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
    """Compute structured financial metrics and a rule-based recommendation.

    Logic used (simple rule-based heuristics):
    - P/E (trailing) if available
    - Revenue growth (from info if available) or None
    - Debt/Equity from info if available
    - Compute PEG if we have earnings growth
    - Recommendation uses multiple signals (PE, PEG, revenue growth, debt/equity)
    """
    symbol = state["symbol"]
    ticker, info, fast, hist = safe_get_ticker(symbol)

    # Extract metrics safely
    trailing_pe = info.get("trailingPE") or fast.get("trailing_pe") or None
    revenue_growth = info.get("revenueGrowth")
    if revenue_growth is not None:
        try:
            revenue_growth_pct = float(revenue_growth) * 100
        except Exception:
            revenue_growth_pct = None
    else:
        revenue_growth_pct = None

    debt_to_equity = info.get("debtToEquity")
    market_cap = info.get("marketCap") or fast.get("market_cap")

    # Attempt PEG using earningsGrowth if provided
    earnings_growth = info.get("earningsQuarterlyGrowth") or info.get("earningsGrowth")
    peg = None
    try:
        if trailing_pe and earnings_growth:
            if isinstance(earnings_growth, (int, float)) and earnings_growth != 0:
                peg = trailing_pe / (earnings_growth * 100)  # if earnings_growth is fraction
            else:
                # If earnings_growth already in percent
                peg = trailing_pe / float(earnings_growth)
    except Exception:
        peg = None

    # Build a metrics dict
    metrics = {
        "P/E (trailing)": trailing_pe if trailing_pe is not None else "N/A",
        "PEG": round(peg, 2) if isinstance(peg, (int, float)) and not math.isnan(peg) else "N/A",
        "Revenue Growth (%)": f"{revenue_growth_pct:.2f}%" if revenue_growth_pct is not None else "N/A",
        "Debt/Equity": round(debt_to_equity, 2) if isinstance(debt_to_equity, (int, float)) else "N/A",
        "Market Cap (USD)": f"${market_cap:,}" if market_cap else "N/A",
    }

    # Rule-based recommendation (expandable)
    recommendation = "Hold"
    try:
        # Conservative buy conditions
        if trailing_pe and trailing_pe < 15 and (revenue_growth_pct is not None and revenue_growth_pct > 10) and (not isinstance(debt_to_equity, (int, float)) or debt_to_equity < 1.5):
            recommendation = "Buy"
        # If PEG shows cheap relative to growth
        elif isinstance(peg, (int, float)) and peg > 0 and peg < 1.5:
            recommendation = "Buy"
        # Clear sell signals
        elif trailing_pe and trailing_pe > 40 and (revenue_growth_pct is not None and revenue_growth_pct < 0):
            recommendation = "Sell"
        else:
            recommendation = "Hold"
    except Exception:
        recommendation = "Hold"

    # Attach derived items to state
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
    """Use the LLM to score sentiment and extract a short rationale.

    If no LLM is available, fall back to keyword heuristics.
    """
    if llm is None:
        # fallback simple heuristic
        desc = (article_description or "").lower()
        positive_keywords = ["beat", "growth", "strong", "positive", "surge", "upgrade", "outperform"]
        negative_keywords = ["miss", "downgrade", "lawsuit", "loss", "recall", "weak", "slow"]
        score = 0
        for w in positive_keywords:
            if w in desc:
                score += 1
        for w in negative_keywords:
            if w in desc:
                score -= 1
        label = "Neutral"
        if score > 0:
            label = "Positive"
        elif score < 0:
            label = "Negative"
        return {"label": label, "score": score, "rationale": "Keyword heuristic"}

    # If LLM is available, ask for a sentiment label and a 1-2 sentence rationale.
    prompt = textwrap.dedent(f"""
    You are a concise financial news grader.
    Given a news headline and short description, return a JSON object with three fields:
    - label: one of Positive, Neutral, Negative
    - score: integer in [-2, -1, 0, 1, 2] where 2 is very positive, -2 very negative
    - rationale: 1-2 sentence explanation in plain English.

    Headline: """ + article_title + """
    Description: """ + (article_description or "") + """

    Return ONLY JSON.
    """)

    response = llm.invoke(prompt)
    # The response format depends on the LLM client; best-effort parsing
    text = getattr(response, "content", str(response))

    # Try to extract JSON-ish content from the response
    import json
    try:
        # naive attempt: find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_text = text[start:end+1]
            parsed = json.loads(json_text)
            return {"label": parsed.get("label", "Neutral"), "score": parsed.get("score", 0), "rationale": parsed.get("rationale", "")}
    except Exception:
        pass

    # fallback if parsing fails
    return {"label": "Neutral", "score": 0, "rationale": text[:300]}


def news_agent(state: FinancialReportState, max_articles: int = 6) -> Dict[str, Any]:
    """Fetch recent news mentioning the symbol and score each article using LLM or heuristic."""
    symbol = state["symbol"]
    if newsapi is None:
        return {"news_sentiment": []}

    try:
        # Query the entire symbol string; for tickers like AAPL this often works
        raw = newsapi.get_everything(q=symbol, language="en", sort_by="relevancy", page_size=max_articles)
        articles = raw.get("articles", [])
    except Exception:
        articles = []

    scored = []
    for a in articles:
        title = a.get("title")
        desc = a.get("description") or a.get("content")
        scored_result = _llm_score_article(title or "", desc or "")
        scored.append({
            "title": title,
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "label": scored_result["label"],
            "score": scored_result["score"],
            "rationale": scored_result["rationale"],
        })

    # Aggregate
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    score_sum = 0
    for s in scored:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
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

def narrative_agent(state: FinancialReportState) -> Dict[str, Any]:
    """Use the LLM to compose a short analyst-style commentary.

    If no LLM is available, fall back to a templated summary.
    """
    today = datetime.date.today().strftime("%B %d, %Y")
    metrics = state.get("financial_metrics") or {}
    agg = state.get("aggregated_sentiment") or {}

    if llm is None:
        # Templated, conservative summary
        lines = [f"{state['symbol']} Investment Commentary - {today}"]
        lines.append("\nSUMMARY:\n")
        lines.append(f"Valuation: {state.get('valuation', 'N/A')}")
        lines.append("Key metrics:\n")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        counts = agg.get('counts', {})
        lines.append(f"News snapshot: {counts.get('Positive',0)} positive, {counts.get('Neutral',0)} neutral, {counts.get('Negative',0)} negative articles.")
        lines.append("\nRecommendation: Monitor upcoming earnings and macro indicators.\n")
        return {"narrative": "\n".join(lines)}

    # Build a careful prompt that provides the model with structure
    prompt = textwrap.dedent(f"""
    You are an experienced equity analyst.
    Today's date: {today}

    Company: {state['symbol']}

    Provide a short (3-5 paragraph) investment commentary suitable for a research note. The commentary should include:
    1) A concise headline (one line) summarizing the recommendation.
    2) A short overview of recent financial metrics (explicitly mention P/E, Revenue Growth, Debt/Equity, PEG if available).
    3) A brief synthesis of recent news (synthesize the aggregated sentiment and 1-2 notable articles with rationale).
    4) A Risks section listing 3 key risks.
    5) A "Next Steps" section suggesting what to monitor next (e.g., earnings, guidance, macro indicators).

    Use the following structured data in your analysis:
    Financial Metrics: {metrics}
    Aggregated News Sentiment: {agg}

    Return plain markdown text; do not include any JSON. Keep the tone formal and concise.
    """)

    response = llm.invoke(prompt)
    narrative_text = getattr(response, "content", str(response))
    return {"narrative": narrative_text}


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

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax[0].plot(history.index, history['Close'])
    ax[0].plot(history.index, history['MA20'])
    ax[0].plot(history.index, history['MA50'])
    # MA200 may be NaN for short series
    ax[0].plot(history.index, history['MA200'])
    ax[0].set_title(f"{symbol} - Close Price with Moving Averages (1y)")
    ax[0].set_ylabel("Price")
    ax[0].grid(True)

    # Volume on the lower axis
    ax[1].bar(history.index, history['Volume'])
    ax[1].set_ylabel("Volume")
    ax[1].set_xlabel("Date")
    ax[1].grid(True)

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return f"![Stock Price Chart](data:image/png;base64,{image_base64})"


def report_agent(state: FinancialReportState) -> Dict[str, Any]:
    today = datetime.date.today().strftime("%B %d, %Y")
    md = [f"# {state['symbol']} Investment Commentary - {today}\n"]

    # Price chart
    hist = state.get('history')
    md.append("## 📈 Price Chart")
    md.append(plot_stock_price_with_indicators(hist, state['symbol']))

    # Key metrics
    md.append("\n## 📊 Key Financial Metrics")
    for k, v in (state.get('financial_metrics') or {}).items():
        md.append(f"- **{k}**: {v}")

    # Valuation badge
    val = state.get('valuation', 'N/A')
    md.append(f"\n**Analyst Recommendation:** **{val}**")

    # News section
    md.append("\n## 📰 News Sentiment Summary")
    agg = state.get('aggregated_sentiment') or {}
    counts = agg.get('counts', {})
    md.append(f"- Positive: {counts.get('Positive', 0)}  |  Neutral: {counts.get('Neutral', 0)}  |  Negative: {counts.get('Negative', 0)}")

    for a in (state.get('news_sentiment') or []):
        md.append(f"- [{a.get('title')}]({a.get('url')}) — {a.get('label')} ({a.get('score')}) — {a.get('rationale')}")

    # Narrative
    md.append("\n## 🧠 Analyst Commentary")
    md.append(state.get('narrative') or "(No narrative generated)")

    # Risks & Next steps placeholders
    md.append("\n## ⚠️ Risks")
    md.append("- Macroeconomic weakness that impacts revenue growth\n- Company-specific execution issues\n- Regulatory or legal risks")

    md.append("\n## ▶️ Next Steps")
    md.append("- Monitor upcoming earnings and management commentary\n- Watch sector ETF and macro indicators (rates, unemployment)\n- Re-run sentiment after major news events")

    report_md = "\n\n".join(md)
    return {"markdown_report": report_md}


# -----------------------------
# Orchestration (sequential runner)
# -----------------------------

def run_financial_report(symbol: str, use_graph: bool = False) -> FinancialReportState:
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

    # Agents executed sequentially (could be replaced by a StateGraph execution)
    state.update(valuation_agent(state))
    state.update(news_agent(state))
    state.update(narrative_agent(state))
    state.update(report_agent(state))

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
            # compiled.run(state)  # If StateGraph provides a run API, this is illustrative
        except Exception:
            pass

    return state


# -----------------------------
# Convenience runner (for notebooks)
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



