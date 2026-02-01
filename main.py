import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# CONFIG: Weekly £80 Allocation
INVESTMENT = 80.00
ALLOCATION = {'VWRP.L': 0.70, 'VUAG.L': 0.30}
DATA_FILE = os.path.join(os.path.dirname(__file__), 'portfolio_data.csv')
CHART_FILE = os.path.join(os.path.dirname(__file__), 'growth_chart.png')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'tracker.log')

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")

try:
    log("Running Weekly Tracker...")
    tickers = list(ALLOCATION.keys())
    data = yf.download(tickers, period="1d")['Adj Close']
    current_prices = {t: float(data[t].iloc[-1]) for t in tickers}
    log(f"Prices Fetched: {current_prices}")

    # Load Database
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        last = df.iloc[-1]
        shares = {'VWRP.L': last['VWRP_Shares'], 'VUAG.L': last['VUAG_Shares']}
        total_invested = last['Total_Invested']
    else:
        df = pd.DataFrame(columns=['Date', 'VWRP_Shares', 'VUAG_Shares', 'Total_Invested', 'Value'])
        shares = {'VWRP.L': 0.0, 'VUAG.L': 0.0}
        total_invested = 0.0

    # Execute "Buy" Logic
    for t in tickers:
        amount = INVESTMENT * ALLOCATION[t]
        new_shares = amount / current_prices[t]
        shares[t] += new_shares
        total_invested += INVESTMENT

    # Calculate Total Value
    current_val = sum(shares[t] * current_prices[t] for t in tickers)

    # Save Data
    new_row = {
        'Date': datetime.now().strftime("%Y-%m-%d"),
        'VWRP_Shares': round(shares['VWRP.L'], 4),
        'VUAG_Shares': round(shares['VUAG.L'], 4),
        'Total_Invested': round(total_invested, 2),
        'Value': round(current_val, 2)
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    log("Database Updated.")

    # Generate Chart
    plt.figure(figsize=(10,6))
    plt.plot(pd.to_datetime(df['Date']), df['Total_Invested'], 'r--', label='Cash In')
    plt.plot(pd.to_datetime(df['Date']), df['Value'], 'g-', linewidth=2, label='Portfolio Value')
    plt.title('Weekly £80 Growth (70% VWRP / 30% VUAG)')
    plt.legend()
    plt.grid(True)
    plt.savefig(CHART_FILE)
    log("Chart Updated.")

except Exception as e:
    log(f"CRITICAL ERROR: {e}")
