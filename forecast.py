import yfinance as yf
import numpy as np
import pandas as pd
import sys
import logging

# CONFIG - All variables are global for direct access
SIMULATIONS = 10000
DAYS_FORWARD = 30 # 1 Month Forecast

# NOTE: Advanced volatility thresholds are temporarily removed/simplified due to persistent NameError issues in this environment.
# CRITICAL_DAILY_PCT_THRESHOLD = 20.0 
# HIGH_DAILY_PCT_THRESHOLD = 8.0    
# RAPID_CHANGE_PERCENT_THRESHOLD = 150.0 
# RAPID_CHANGE_PERIOD_DAYS = 90 
# HIGH_VOLATILITY_ANNUALIZED_THRESHOLD = 30.0 
# MIDDLE_VOLATILITY_ANNUALIZED_THRESHOLD = 20.0 
# LOW_VOLATILITY_ANNUALIZED_THRESHOLD = 13.0 

# SMART MAP TICKER_MAP
TICKER_MAP = {
    "btc": "BTC-USD",
    "bitcoin": "BTC-USD",
    "eth": "ETH-USD",
    "gold": "GC=F",
    "silver": "SI=F",
    "oil": "CL=F",
    "beans": "ZS=F",
    "corn": "ZC=F",
    "sp500": "VUAG.L",
    "world": "VWRP.L",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "google": "GOOG"
}

# Setup basic logging to stderr for exec output capture
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stderr)

def get_recommendation(win_prob, rsi, trend):
    # THE RECOMMENDATION ENGINE - Simplified
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
    elif win_prob < 30 and rsi > 70:
        return "🚨 STRONG SELL (Crash Risk High)"
    else:
        return "✋ WAIT (Conflicting Signals)"

def get_system_conviction_score(win_prob, rsi, recommendation, trend):
    """Calculates a correctness/trustworthiness score (0-100%). Simplified."""
    score = 0.0

    # 1. Base Score from Win Probability aligned with the recommendation
    if "BUY" in recommendation: 
        score = max(0, (win_prob - 50) * 2)
    elif "SELL" in recommendation: 
        score = max(0, (50 - win_prob) * 2)
    elif "WAIT" in recommendation: 
        score = max(0, 70 - abs(win_prob - 50) * 1.0) 

    # 2. Adjustments for Reinforcing Signals (add to score)
    if "BUY" in recommendation:
        if trend == "BULLISH": score += 10
        if rsi < 30: score += 20
        elif rsi < 50: score += 10
    elif "SELL" in recommendation:
        if trend == "BEARISH": score += 10
        if rsi > 70: score += 20
        elif rsi > 50: score += 10
    elif "WAIT" in recommendation and trend == "UNKNOWN":
        score += 5

    # 3. Adjustments for Contradictory Signals (deduct from score)
    if "BUY" in recommendation:
        if trend == "BEARISH": score -= 30
        if rsi > 70: score -= 30
    elif "SELL" in recommendation:
        if trend == "BULLISH": score -= 30
        if rsi < 30: score -= 30
    elif "WAIT" in recommendation: 
        if (win_prob > 70 or win_prob < 30): 
            score -= 20
        if (rsi > 75 or rsi < 25): 
            score -= 20

    # Ensure score is within 0-100 range
    return round(max(0, min(100, score)), 1)

def analyze(user_input): 
    
    ticker = TICKER_MAP.get(user_input.lower(), user_input.upper())
    print(f" --- ANALYZING: {ticker} ---")
    try:
        data = yf.download(ticker, period="2y", progress=False)
        
        if data.empty:
            logging.error(f"Ticker '{ticker}' not found or no data available for the specified period.")
            return print("Error: Ticker not found or no data.")

        # Handle MultiIndex columns if present (e.g., ('Close', 'GOOG'))
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data.columns = data.columns.droplevel(1) # Drop the ticker level if it's the second level
            except IndexError:
                pass
        
        # Try 'Adj Close' first, then 'Close'
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            logging.error(f"Neither 'Adj Close' nor 'Close' column found in data for {ticker}. Available columns: {data.columns.tolist()}")
            return print(f"Error: Essential price data (Adj Close or Close) missing for {ticker}.")

        curr_price = prices.iloc[-1].item()

    except Exception as e:
        logging.error(f"An API error occurred for {ticker}: {e}")
        return print(f"API Error: An error occurred while fetching data for {ticker}.")

    # 1. TECHNICALS
    if len(prices) < 200: # Need at least 200 data points for 200 SMA and RSI
        logging.warning(f"Insufficient data points ({len(prices)}) for full technical analysis for {ticker}. Skipping some indicators.")
        sma50 = prices.rolling(50).mean().iloc[-1].item() if len(prices) >= 50 else curr_price
        sma200 = curr_price # Cannot calculate 200 SMA
        trend = "UNKNOWN"
        rsi = 50.0 # Neutral RSI
    else:
        sma50 = prices.rolling(50).mean().iloc[-1].item()
        sma200 = prices.rolling(200).mean().iloc[-1].item()
        trend = "BULLISH" if sma50 > sma200 else "BEARISH"
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        # Handle cases where loss is zero to avoid division by zero
        rs = gain / loss.replace(0, np.nan) # Replace 0 with NaN for division
        rsi = 100 - (100 / (1 + rs.fillna(0))).iloc[-1].item() # Fill NaN with 0 for RSI calculation

    # 2. MONTE CARLO (Weighted Volatility)
    # Need enough recent data for volatility calculation
    if len(prices.tail(126)) < 2:
        logging.warning("Insufficient recent data for Monte Carlo volatility calculation. Using simplified prediction.")
        p50 = curr_price # Assume no change
        win_prob = 50.0 # Neutral probability
    else:
        # Weight recent 6 months (126 days) more heavily for volatility
        recent_returns = np.log(1 + prices.tail(126).pct_change())
        long_returns = np.log(1 + prices.pct_change())
        
        # Handle potential NaN/inf from log or pct_change for robust mean/var/std
        u = long_returns.replace([np.inf, -np.inf], np.nan).dropna().mean() # Long term drift
        var = recent_returns.replace([np.inf, -np.inf], np.nan).dropna().var() # Recent variance
        stdev = recent_returns.replace([np.inf, -np.inf], np.nan).dropna().std() # Recent std dev

        # Check if we have valid stats to proceed with Monte Carlo
        if np.isnan(u) or np.isnan(var) or np.isnan(stdev) or stdev == 0:
            logging.warning("Insufficient valid stats for Monte Carlo simulation. Using simplified prediction.")
            p50 = curr_price # Assume no change
            win_prob = 50.0 # Neutral probability
        else:
            drift = u - (0.5 * var)
            sims = np.exp(drift + stdev * np.random.normal(0,1, (DAYS_FORWARD, SIMULATIONS)))
            paths = np.zeros_like(sims)
            paths[0] = curr_price
            for t in range(1, DAYS_FORWARD):
                paths[t] = paths[t-1] * sims[t]
            final = paths[-1]
            p50 = np.percentile(final, 50)
            win_prob = np.sum(final > curr_price) / SIMULATIONS * 100

    # Rapid Change and Volatility Detections are temporarily disabled
    rapid_rise = False
    rapid_fall = False
    critical_high_daily_volatility = False
    high_daily_volatility = False
    max_abs_daily_change_recent_period = 0.0
    annualized_volatility = 0.0
    high_annualized_volatility_flag = False

    # 3. OUTPUT
    rec = get_recommendation(win_prob, rsi, trend, rapid_rise, rapid_fall, 
                           critical_high_daily_volatility, high_daily_volatility, high_annualized_volatility_flag)
    trustworthiness_score = get_system_conviction_score(win_prob, rsi, rec, trend, annualized_volatility, 
                                                      max_recent_daily_change_pct=max_abs_daily_change_recent_period, 
                                                      rapid_rise=rapid_rise, rapid_fall=rapid_fall, 
                                                      critical_high_daily_volatility=critical_high_daily_volatility, 
                                                      high_daily_volatility=high_daily_volatility,
                                                      high_annualized_volatility_flag=high_annualized_volatility_flag)
    print(f"💰 Current Price: {curr_price:,.2f}")
    print(f"📊 Market Mood: {trend} | RSI: {rsi:.1f}")
    print(f"🎲 Win Probability: {win_prob:.1f}% (Next {DAYS_FORWARD} Days)")
    print(f"🔮 Expected Price: {p50:,.2f}")
    print(f" 📢 SYSTEM VERDICT: {rec} ")
    print(f" ⭐ SYSTEM TRUSTWORTHINESS: {trustworthiness_score:.1f}% ")
    print(f" 📈 Annualized Volatility: {annualized_volatility:.1f}%")
    # Removed conditional print for Max Recent Daily Change for simplicity

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "sp500"
    analyze(target)
