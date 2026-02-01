import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys
import argparse

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def get_next_trading_day(date_obj, data_prices):
    """Attempts to find the next actual trading day in the historical data."""
    current_date_str = date_obj.strftime('%Y-%m-%d')
    try:
        idx = data_prices.index.get_loc(current_date_str)
        # Ensure there is a next day
        if idx + 1 < len(data_prices.index):
            next_trading_day = data_prices.index[idx + 1]
            return next_trading_day.to_pydatetime().date()
        else:
            logging.warning(f"No more trading days in fetched data after {current_date_str}.")
            return None
    except KeyError:
        logging.warning(f"Date {current_date_str} not found in trading days or is last day of data. Trying next calendar day...")
        candidate_date = date_obj + timedelta(days=1)
        for _ in range(5): 
            if candidate_date.weekday() < 5: # Assume next weekday is a trading day
                return candidate_date
            candidate_date += timedelta(days=1)
        logging.error(f"Could not find a valid next trading day after {date_obj.strftime('%Y-%m-%d')}")
        return None
    except IndexError: 
        logging.warning(f"{current_date_str} is the last day in historical data. Cannot determine next trading day from data. Trying next calendar day...")
        candidate_date = date_obj + timedelta(days=1)
        for _ in range(5):
            if candidate_date.weekday() < 5: # Assume next weekday is a trading day
                return candidate_date
            candidate_date += timedelta(days=1)
        return None

def predict_next_day_movement(ticker, date_for_prediction, short_ma=5, long_ma=20, rsi_period=14, full_historical_prices=None):
    """
    Predicts if the next day's close price will be 'UP', 'DOWN', or 'UNSURE'
    based on historical data up to (and including) date_for_prediction.
    
    Args:
        ticker (str): Stock ticker symbol.
        date_for_prediction (datetime.date): The date for which to use data to make the prediction.
                                            The prediction will be for the *next trading day* after this date.
        full_historical_prices (pd.Series, optional): Pre-fetched historical prices to avoid repeated downloads.
                                                    If None, data will be downloaded.
    Returns:
        tuple: (prediction_str, actual_movement_str, correctness_boolean)
    """
    logging.info(f"--- Starting Prediction for {ticker} for day after {date_for_prediction.strftime('%Y-%m-%d')} ---")

    prices = None
    if full_historical_prices is not None:
        prices = full_historical_prices
    else:
        # Fetch data up to (and including) the prediction date to calculate indicators
        # Fetch a wider period to ensure enough data for MAs and RSI
        start_date_fetch = date_for_prediction - timedelta(days=730) # Get 2 years history
        end_date_fetch = date_for_prediction + timedelta(days=30) # Include a buffer for actual comparison

        try:
            data = yf.download(ticker, start=start_date_fetch, end=end_date_fetch, progress=False)
            if data.empty:
                logging.error(f"No data found for {ticker} between {start_date_fetch.strftime('%Y-%m-%d')} and {end_date_fetch.strftime('%Y-%m-%d')}. ")
                return "N/A", "N/A", False

            # Flatten MultiIndex if necessary
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Ensure we have 'Close' or 'Adj Close'
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            if price_col not in data.columns:
                logging.error(f"Essential price data ({price_col}) missing for {ticker}. Available columns: {data.columns.tolist()}")
                return "N/A", "N/A", False

            prices = data[price_col]

        except Exception as e:
            logging.error(f"An error occurred during data download for {ticker}: {e}")
            return "ERROR", "ERROR", False

    # Get data up to the prediction date for indicator calculation
    historical_prices = prices.loc[:date_for_prediction.strftime('%Y-%m-%d')].dropna()
    if len(historical_prices) < long_ma + rsi_period: # Need enough data for indicators
        logging.warning(f"Insufficient historical data for {ticker} up to {date_for_prediction.strftime('%Y-%m-%d')} for full indicator calculation. Only {len(historical_prices)} data points.")
        return "UNSURE", "N/A", False

    # 1. Calculate Indicators using historical_prices
    sma_short = historical_prices.rolling(window=short_ma).mean().iloc[-1]
    sma_long = historical_prices.rolling(window=long_ma).mean().iloc[-1]
    
    delta = historical_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan) # Avoid division by zero
    rsi = 100 - (100 / (1 + rs.fillna(0))).iloc[-1]

    # 2. Prediction Logic
    prediction_score = 0
    if sma_short > sma_long: prediction_score += 1 # Bullish trend
    if sma_short < sma_long: prediction_score -= 1 # Bearish trend
    if rsi > 55: prediction_score += 1 # Strong momentum up
    if rsi < 45: prediction_score -= 1 # Strong momentum down
    if rsi < 30: prediction_score += 1 # Oversold, potential bounce
    if rsi > 70: prediction_score -= 1 # Overbought, potential correction
    
    if prediction_score > 0:
        prediction_str = "UP"
    elif prediction_score < 0:
        prediction_str = "DOWN"
    else:
        prediction_str = "UNSURE"
    
    logging.info(f"Indicators for {ticker} on {date_for_prediction.strftime('%Y-%m-%d')}: SMA_S={sma_short:.2f}, SMA_L={sma_long:.2f}, RSI={rsi:.1f}. Score={prediction_score}")

    # 3. Determine Actual Movement for correctness check
    actual_movement_str = "N/A"
    correctness_boolean = False
    
    # Find the next actual trading day after date_for_prediction
    actual_start_date = date_for_prediction + timedelta(days=1)
    future_prices = prices.loc[actual_start_date.strftime('%Y-%m-%d'):].dropna()

    if not future_prices.empty:
        predicted_day_actual_date = future_prices.index[0].to_pydatetime().date()
        last_price_for_prediction = historical_prices.iloc[-1]
        actual_close_price = future_prices.iloc[0]

        if actual_close_price > last_price_for_prediction:
            actual_movement_str = "UP"
        elif actual_close_price < last_price_for_prediction:
            actual_movement_str = "DOWN"
        else:
            actual_movement_str = "NO CHANGE"
        
        correctness_boolean = (prediction_str == actual_movement_str)
        logging.info(f"Actual movement on {predicted_day_actual_date.strftime('%Y-%m-%d')}: {actual_movement_str} (Prev Close: {last_price_for_prediction:.2f}, Actual Close: {actual_close_price:.2f})")
        logging.info(f"Prediction Correct: {correctness_boolean}")
    else:
        logging.warning(f"No actual trading data found after {date_for_prediction.strftime('%Y-%m-%d')} to verify prediction.")

    return prediction_str, actual_movement_str, correctness_boolean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Next-day stock movement predictor.")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., VWRP.L)")
    parser.add_argument("--date", type=str, help="Specific date for prediction (YYYY-MM-DD). If omitted, runs batch backtest.", default=None)
    parser.add_argument("--test-days", type=int, help="Number of historical test days for batch backtesting.", default=0)
    args = parser.parse_args()

    target_ticker = args.ticker
    
    if args.date:
        try:
            prediction_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logging.error("Invalid date format. Please use YYYY-MM-DD.")
            sys.exit(1)

        predicted_move, actual_move, is_correct = predict_next_day_movement(target_ticker, prediction_date)
        print(f"\n--- SINGLE BACKTEST RESULT for {target_ticker} ---")
        print(f"Date used for prediction (indicators calculated on): {prediction_date.strftime('%Y-%m-%d')}")
        print(f"Predicted Movement for next trading day: {predicted_move}")
        print(f"Actual Movement on next trading day: {actual_move}")
        print(f"Prediction was Correct: {is_correct}")

    elif args.test_days > 0:
        logging.info(f"--- Starting Batch Backtest for {target_ticker} ({args.test_days} days) ---")
        # Fetch 3 years of data to ensure enough trading days for sampling
        full_data_start = datetime.now().date() - timedelta(days=3 * 365)
        full_data_end = datetime.now().date()
        try:
            full_historical_data = yf.download(target_ticker, start=full_data_start, end=full_data_end, progress=False)
            if full_historical_data.empty:
                logging.error(f"Could not download sufficient historical data for batch backtest for {target_ticker}.")
                sys.exit(1)
            if isinstance(full_historical_data.columns, pd.MultiIndex):
                full_historical_data.columns = full_historical_data.columns.get_level_values(0)
            price_col = 'Adj Close' if 'Adj Close' in full_historical_data.columns else 'Close'
            full_prices = full_historical_data[price_col].dropna()
            if len(full_prices) < 200: # Ensure enough data for indicators on any given day
                logging.error(f"Insufficient historical data (less than 200 trading days) for reliable batch backtest for {target_ticker}.")
                sys.exit(1)

        except Exception as e:
            logging.error(f"Error fetching full historical data for batch backtest: {e}")
            sys.exit(1)

        trading_dates = full_prices.index.to_pydatetime() # Get all trading dates
        # Filter out very recent dates to ensure we have actual next-day data for comparison
        # Keep dates where there's at least one subsequent trading day in full_prices
        valid_prediction_dates = []
        for i in range(len(trading_dates) - 1):
            # Ensure the prediction date itself has enough prior history for indicators
            # We need at least (long_ma + rsi_period) days prior to the prediction date.
            # For simplicity, filter out dates too early in the full_prices set
            if i >= 250: # Roughly 1 year of data to compute indicators
                valid_prediction_dates.append(trading_dates[i].date())
        
        if len(valid_prediction_dates) < args.test_days:
            logging.warning(f"Only {len(valid_prediction_dates)} valid prediction dates available, requested {args.test_days}. Using all available.")
            sample_dates = valid_prediction_dates
        else:
            # Randomly sample dates for even distribution across trading days
            np.random.seed(42) # For reproducibility
            sample_dates = np.random.choice(valid_prediction_dates, args.test_days, replace=False)
            sample_dates = sorted([d for d in sample_dates]) # Sort dates for chronological order

        results = []
        for date_to_test in sample_dates:
            # Pass full_prices to avoid re-downloading data for each prediction
            predicted_move, actual_move, is_correct = predict_next_day_movement(target_ticker, date_to_test, full_historical_prices=full_prices)
            results.append({
                "date": date_to_test.strftime('%Y-%m-%d'),
                "predicted": predicted_move,
                "actual": actual_move,
                "correct": is_correct
            })
        
        correct_count = sum(1 for r in results if r['correct'] is True)
        total_predictions = len(results)
        accuracy = (correct_count / total_predictions) * 100 if total_predictions > 0 else 0

        print(f"\n--- BATCH BACKTEST SUMMARY for {target_ticker} ({total_predictions} Predictions) ---")
        print(f"Total Correct: {correct_count}")
        print(f"Total Incorrect: {total_predictions - correct_count}")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print("\n--- Individual Results ---")
        for r in results:
            print(f"Date: {r['date']} | Pred: {r['predicted']} | Actual: {r['actual']} | Correct: {r['correct']}")

    else:
        logging.error("Please provide either --date YYYY-MM-DD for a single backtest or --test-days N for a batch backtest.")
        sys.exit(1)
