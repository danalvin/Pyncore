#!/usr/bin/env python3
"""
PyneCore Direct Usage Example - CWR Indicator
This demonstrates how to use PyneCore directly from Python without the CLI
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))


# Create a sample PyneCore script file
def create_pyne_script():
    """Create a PyneCore script file for the CWR indicator"""

    script_content = '''"""
@pyne
"""
from pynecore import Series, Persistent
from pynecore.lib import script, high, low, close, ta, plot, color, input, na


@script.indicator(title="Candle Widening Ratio (CWR)")
def main(
    length: int = input.int("Length", 20, minval=1, maxval=200),
    show_thresholds: bool = input.bool("Show Thresholds", True),
    high_threshold: float = input.float("High Threshold", 1.5, minval=0.1, step=0.1),
    low_threshold: float = input.float("Low Threshold", 0.5, minval=0.1, step=0.1)
):
    """
    Calculate the Candle Widening Ratio (CWR) indicator.

    CWR = (current_range) / (average_range_of_last_N_candles)
    where current_range = high - low
    """

    # Calculate current candle range
    current_range = high - low

    # Calculate average range of the last N candles using Simple Moving Average
    avg_range = ta.sma(current_range, length)

    # Calculate CWR: current range / average range
    # Handle division by zero case
    cwr = na if avg_range == 0 else current_range / avg_range

    # Plot the main CWR line
    plot(cwr, title="CWR", color=color.blue, linewidth=2)

    # Plot threshold lines if enabled
    if show_thresholds:
        plot(1.0, title="Baseline", color=color.gray, linewidth=1)
        plot(high_threshold, title="High Threshold", color=color.red, linewidth=1)
        plot(low_threshold, title="Low Threshold", color=color.green, linewidth=1)

    # Additional analysis: detect volatility spikes
    high_volatility = cwr > high_threshold
    low_volatility = cwr < low_threshold

    # Count for statistics (using persistent variables)
    high_vol_count: Persistent[int] = 0
    low_vol_count: Persistent[int] = 0
    total_bars: Persistent[int] = 0

    if not na(cwr):
        total_bars += 1
        if high_volatility:
            high_vol_count += 1
        if low_volatility:
            low_vol_count += 1

    # Return the CWR value for external use
    return cwr


# Additional utility functions can be defined here
def get_cwr_stats():
    """Get CWR statistics - this would be called after processing"""
    pass
'''

    with open('cwr_indicator.py', 'w') as f:
        f.write(script_content)

    return 'cwr_indicator.py'


def generate_sample_ohlcv_data(filename: str = "sample_data.ohlcv", num_bars: int = 500):
    """
    Generate sample OHLCV data in PyneCore's .ohlcv format

    The .ohlcv format is a binary format used by PyneCore for efficient data storage.
    For this example, we'll create a CSV and convert it conceptually.
    """

    np.random.seed(42)

    # Generate realistic price data
    base_price = 50000
    volatility = np.random.Generator(0.01, 0.03, num_bars)
    returns = np.random.Generator(0, 1, num_bars) * volatility

    # Add some trend
    trend = np.sin(np.arange(num_bars) * 0.01) * 0.0005
    returns = returns + trend

    # Generate price series
    close_prices = [base_price]
    for i in range(1, num_bars):
        new_price = close_prices[-1] * (1 + returns[i])
        close_prices.append(max(new_price, 100))

    # Generate OHLC data
    data = []
    for i in range(num_bars):
        close_price = close_prices[i]

        # Generate range
        range_pct = np.random.Generator(0.005, 0.04) * (1 + volatility[i] * 3)
        range_value = close_price * range_pct

        # Generate OHLC
        if i == 0:
            open_price = close_price + np.random.Generator(-range_value / 4, range_value / 4)
        else:
            gap = np.random.Generator(0, 0.001) * close_prices[i - 1]
            open_price = close_prices[i - 1] + gap

        high_price = max(open_price, close_price) + np.random.Generator(0, range_value / 2)
        low_price = min(open_price, close_price) - np.random.Generator(0, range_value / 2)

        # Ensure logical order
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Generate volume
        volume = np.random.Generator(1000, 5000)

        # Create timestamp (assuming 1-hour bars)
        timestamp = datetime.now() - timedelta(hours=(num_bars - 1 - i))

        data.append({
            'timestamp': timestamp.isoformat(),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2)
        })

    # Save as CSV (PyneCore can convert this to .ohlcv format)
    df = pd.DataFrame(data)
    csv_filename = filename.replace('.ohlcv', '.csv')
    df.to_csv(csv_filename, index=False)

    print(f"Generated {num_bars} bars of sample data")
    print(f"Saved to {csv_filename}")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

    return csv_filename


def simulate_pynecore_execution(data_path: str):
    """
    Simulate PyneCore execution directly from Python

    This is a conceptual implementation since PyneCore's internal execution
    is complex and involves AST transformations.
    """

    print("=" * 50)
    print("SIMULATING PYNECORE EXECUTION")
    print("=" * 50)

    # Load the data
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found!")
        return None

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Loaded {len(df)} bars from {data_path}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Simulate the CWR calculation (this is what PyneCore would do internally)
    length = 20
    current_range = df['high'] - df['low']
    avg_range = current_range.rolling(window=length, min_periods=1).mean()
    cwr = current_range / avg_range

    # Handle NaN values
    cwr = cwr.fillna(0)

    # Add to dataframe
    df['cwr'] = cwr

    # Calculate statistics (simulating PyneCore's persistent variables)
    high_threshold = 1.5
    low_threshold = 0.5

    high_vol_count = (cwr > high_threshold).sum()
    low_vol_count = (cwr < low_threshold).sum()
    total_bars = len(df)

    # Create results
    results = {
        'indicator': 'Candle Widening Ratio (CWR)',
        'parameters': {
            'length': length,
            'high_threshold': high_threshold,
            'low_threshold': low_threshold
        },
        'statistics': {
            'total_bars': total_bars,
            'high_volatility_count': int(high_vol_count),
            'low_volatility_count': int(low_vol_count),
            'high_volatility_percentage': float(high_vol_count / total_bars * 100),
            'low_volatility_percentage': float(low_vol_count / total_bars * 100),
            'mean_cwr': float(cwr.mean()),
            'std_cwr': float(cwr.std()),
            'min_cwr': float(cwr.min()),
            'max_cwr': float(cwr.max())
        },
        'recent_values': cwr.tail(10).tolist(),
        'data': df.to_dict('records')
    }

    print("\nCWR Statistics:")
    print(f"  Mean: {results['statistics']['mean_cwr']:.4f}")
    print(f"  Std: {results['statistics']['std_cwr']:.4f}")
    print(f"  Min: {results['statistics']['min_cwr']:.4f}")
    print(f"  Max: {results['statistics']['max_cwr']:.4f}")
    print(
        f"  High Volatility: {results['statistics']['high_volatility_count']} ({results['statistics']['high_volatility_percentage']:.1f}%)")
    print(
        f"  Low Volatility: {results['statistics']['low_volatility_count']} ({results['statistics']['low_volatility_percentage']:.1f}%)")

    return results


def save_results_to_files(results: dict):
    """Save results to JSON and CSV files"""

    # Save complete results to JSON
    with open('cwr_pynecore_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("✓ Complete results saved to cwr_pynecore_results.json")

    # Save data with CWR to CSV
    df_data = pd.DataFrame(results['data'])
    df_data.to_csv('cwr_pynecore_data.csv', index=False)
    print("✓ Data with CWR values saved to cwr_pynecore_data.csv")

    # Save summary statistics
    summary = {
        'indicator': results['indicator'],
        'parameters': results['parameters'],
        'statistics': results['statistics'],
        'timestamp': datetime.now().isoformat()
    }

    with open('cwr_pynecore_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("✓ Summary statistics saved to cwr_pynecore_summary.json")


def main():
    """
    Main function demonstrating direct PyneCore usage
    """

    print("PyneCore Direct Usage - CWR Indicator")
    print("=" * 50)

    # Step 1: Create the PyneCore script
    print("1. Creating PyneCore script...")
    script_path = create_pyne_script()
    print(f"   Created: {script_path}")

    # Step 2: Generate sample data
    print("\n2. Generating sample OHLCV data...")
    data_path = generate_sample_ohlcv_data("sample_data.csv", 500)
    print(f"   Created: {data_path}")

    # Step 3: Simulate PyneCore execution
    print("\n3. Simulating PyneCore execution...")
    results = simulate_pynecore_execution(data_path)

    if results:
        # Step 4: Save results
        print("\n4. Saving results...")
        save_results_to_files(results)

        print("\n" + "=" * 50)
        print("EXECUTION COMPLETE")
        print("=" * 50)
        print("Files generated:")
        print("  • cwr_indicator.py - PyneCore script")
        print("  • sample_data.csv - Sample OHLCV data")
        print("  • cwr_pynecore_results.json - Complete results")
        print("  • cwr_pynecore_data.csv - Data with CWR values")
        print("  • cwr_pynecore_summary.json - Summary statistics")

        print("\nTo use with real PyneCore:")
        print("  1. Install PyneCore: pip install pynesys-pynecore[cli]")
        print("  2. Convert CSV to .ohlcv: pyne data convert-from sample_data.csv --symbol BTCUSDT --timeframe 1h")
        print("  3. Run the script: pyne run cwr_indicator.py sample_data.ohlcv")
    else:
        print("Execution failed!")


if __name__ == "__main__":
    main()