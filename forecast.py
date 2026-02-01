import yfinance as yf
import numpy as np
import pandas as pd
import sys
import logging

# CONFIG
SIMULATIONS = 10000
DAYS_FORWARD = 30 

TICKER_MAP = {
    "btc": "BTC-USD",
    "bitcoin": "BTC-USD",
    "sp500": "VUAG.L",
    "world": "VWRP.L",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "google": "GOOG"
}

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stderr)

def get_recommendation(win_prob, rsi, trend):
    """Simplified Recommendation Engine"""
    if win_prob > 65 and rsi < 35:
        return "🚀 STRONG BUY (High Probability + Oversold)"
    elif win_prob > 60 and rsi < 70:
        return "✅ BUY (Positive Trend)"
    elif win_prob > 40 and win_prob <= 60:
        return "✋ WAIT / HOLD (Market is Uncertain)"
    elif rsi > 75:
        return "⚠️ WAIT (Market Overheated/Overbought)"
    elif win_prob < 40 and trend == "BEARISH":
        return "❌ SELL (Negative Trend)"
    else:
        return "✋ WAIT (Conflicting Signals)"

def get_system_conviction_score(win_prob, rsi, recommendation, trend):
    """Calculates a trust score 0-100%"""
    score = 50.0 # Base score
    if "BUY" in recommendation and trend == "BULLISH": score += 20
    if "SELL" in recommendation and trend == "BEARISH": score += 20
    if rsi < 30 or rsi > 70: score += 10 # Stronger signal if RSI is at extremes
    
    return round(max(0, min(100, score)), 1)

def analyze(user_input): 
    ticker = TICKER_MAP.get(user_input.lower(), user_input.upper())
    print(f"--- ANALYZING: {ticker} ---")
    
    try:
        data = yf.download(ticker, period="2y", progress=False)
        if data.empty:
            print(f"Error: No data found for {ticker}")
            return

        # Fix for Yahoo Finance MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        curr_price = float(prices.iloc[-1])

        # 1. TECHNICALS
        sma50 = prices.rolling(50).mean().iloc[-1]
        sma200 = prices.rolling(200).mean().iloc[-1]
        trend = "BULLISH" if sma50 > sma200 else "BEARISH"
        
        # RSI Calculation
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs.fillna(0))).iloc[-1]

        # 2. MONTE CARLO
        returns = np.log(1 + prices.pct_change().dropna())
        u = returns.mean()
        var = returns.var()
        stdev = returns.std()
        drift = u - (0.5 * var)
        
        # Simulation
        daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (DAYS_FORWARD, SIMULATIONS)))
        path = np.zeros_like(daily_returns)
        path[0] = curr_price
        for t in range(1, DAYS_FORWARD):
            path[t] = path[t-1] * daily_returns[t]
        
        p50 = np.percentile(path[-1], 50)
        win_prob = (np.sum(path[-1] > curr_price) / SIMULATIONS) * 100

        # 3. OUTPUT - FIXED CALLS
        rec = get_recommendation(win_prob, rsi, trend)
        score = get_system_conviction_score(win_prob, rsi, rec, trend)

        print(f"💰 Price: {curr_price:,.2f} | Mood: {trend} | RSI: {rsi:.1f}")
        print(f"🎲 Win Prob: {win_prob:.1f}% | Forecast: {p50:,.2f}")
        print(f"📢 VERDICT: {rec}")
        print(f"⭐ TRUST: {score}%")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "sp500"
    analyze(target)
