"""
Generate sample OHLCV data in CSV format for testing the CWR indicator
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_ohlcv_csv(filename: str = "ohlcv_data.csv", num_bars: int = 500):
    """
    Generate realistic OHLCV data and save to CSV
    """
    np.random.seed(42)

    # Start with a base price (e.g., BTC-like)
    base_price = 45000

    # Generate realistic price movements with some volatility clustering
    volatility = np.random.Generator(0.01, 0.04, num_bars)  # Varying volatility
    returns = np.random.Generator(0, 1, num_bars) * volatility

    # Add some trend and mean reversion
    trend = np.sin(np.arange(num_bars) * 0.02) * 0.001
    returns = returns + trend

    # Create price series
    close_prices = [base_price]
    for i in range(1, num_bars):
        new_price = close_prices[-1] * (1 + returns[i])
        close_prices.append(max(new_price, 100))  # Ensure positive prices

    # Generate OHLC data
    data = []
    for i in range(num_bars):
        close_price = close_prices[i]

        # Generate realistic range based on volatility
        range_pct = np.random.Generator(0.005, 0.08) * (1 + volatility[i] * 10)
        range_value = close_price * range_pct

        # Generate open price (correlated with previous close)
        if i == 0:
            open_price = close_price + np.random.Generator(-range_value / 4, range_value / 4)
        else:
            # Open is usually close to previous close with some gap
            gap = np.random.Generator(0, 0.002) * close_prices[i - 1]
            open_price = close_prices[i - 1] + gap

        # Generate high and low
        high_price = max(open_price, close_price) + np.random.Generator(0, range_value / 3)
        low_price = min(open_price, close_price) - np.random.Generator(0, range_value / 3)

        # Ensure logical order
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Generate volume (higher volume with larger price moves)
        price_change = abs(close_price - open_price) / open_price
        base_volume = np.random.Generator(500, 2000)
        volume = base_volume * (1 + price_change * 10)

        # Create timestamp (1-hour intervals)
        timestamp = datetime.now() - timedelta(hours=(num_bars - 1 - i))

        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2)
        })

    # Create DataFrame and save
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Parse timestamp column

    # Ensure strictly increasing timestamps
    df = df.sort_values('timestamp')

    # Optional: Deduplicate
    df = df.drop_duplicates(subset='timestamp')

    # Save
    df.to_csv(filename, index=False)


    print(f"Generated {num_bars} bars of OHLCV data")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"Average volume: {df['volume'].mean():.2f}")
    print(f"Data saved to {filename}")

    return df


if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_ohlcv_csv("ohlcv_data.csv", 500)

    # Display first few rows
    print("\nFirst 5 rows of generated data:")
    print(df.head())

    # Display last few rows
    print("\nLast 5 rows of generated data:")
    print(df.tail())