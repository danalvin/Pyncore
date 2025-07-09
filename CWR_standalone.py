#!/usr/bin/env python3
"""
Candle Widening Ratio (CWR) Indicator - Standalone Version
This version works independently without requiring PyneCore installation
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available. Install with: pip install matplotlib")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not available. Install with: pip install plotly")


class CandleWideningRatio:
    """
    Candle Widening Ratio (CWR) Indicator

    Measures how the current candle's range (high - low) compares to the
    average range of the previous N candles.

    Formula: CWR = (current_range) / (average_range_of_last_N_candles)
    """

    def __init__(self, length: int = 20):
        """
        Initialize the CWR calculator

        Args:
            length: Number of previous candles to use for average calculation
        """
        self.length = length
        self.name = "Candle Widening Ratio"
        self.short_name = "CWR"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate CWR for OHLCV dataframe

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Series with CWR values
        """
        # Validate input
        required_columns = ['high', 'low']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate current range for each candle
        current_range = df['high'] - df['low']

        # Calculate rolling average of range over the last N candles
        avg_range = current_range.rolling(window=self.length, min_periods=1).mean()

        # Calculate CWR: current range / average range
        cwr = current_range / avg_range

        # Handle division by zero (should be rare with real data)
        cwr = cwr.replace([np.inf, -np.inf], np.nan)

        return cwr

    def get_signals(self, cwr_values: pd.Series, high_threshold: float = 1.5, low_threshold: float = 0.5) -> Dict[
        str, pd.Series]:
        """
        Generate trading signals based on CWR values

        Args:
            cwr_values: Series of CWR values
            high_threshold: Threshold for high volatility signal
            low_threshold: Threshold for low volatility signal

        Returns:
            Dictionary with signal series
        """
        signals = {
            'high_volatility': cwr_values > high_threshold,
            'low_volatility': cwr_values < low_threshold,
            'normal_volatility': (cwr_values >= low_threshold) & (cwr_values <= high_threshold),
            'expanding_range': cwr_values > 1.0,
            'contracting_range': cwr_values < 1.0
        }

        return signals

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of CWR

        Args:
            df: OHLCV dataframe

        Returns:
            Analysis results dictionary
        """
        cwr_values = self.calculate(df)
        signals = self.get_signals(cwr_values)

        # Calculate statistics
        stats = {
            'mean': float(cwr_values.mean()),
            'std': float(cwr_values.std()),
            'min': float(cwr_values.min()),
            'max': float(cwr_values.max()),
            'median': float(cwr_values.median()),
            'skewness': float(cwr_values.skew()),
            'kurtosis': float(cwr_values.kurtosis())
        }

        # Signal frequencies
        signal_stats = {}
        for signal_name, signal_series in signals.items():
            signal_stats[signal_name] = {
                'count': int(signal_series.sum()),
                'frequency': float(signal_series.mean()),
                'percentage': float(signal_series.mean() * 100)
            }

        # Recent values (last 10)
        recent_values = cwr_values.tail(10).to_dict()

        return {
            'statistics': stats,
            'signals': signal_stats,
            'recent_values': recent_values,
            'total_bars': len(df),
            'length_parameter': self.length
        }


def generate_sample_data(num_bars: int = 500) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing
    """
    np.random.seed(42)

    # Parameters for realistic data
    base_price = 45000

    # Generate price movements with volatility clustering
    volatility = np.random.Generator(0.01, 0.04, num_bars)
    returns = np.random.Generator(0, 1, num_bars) * volatility

    # Add trend component
    trend = np.sin(np.arange(num_bars) * 0.02) * 0.001
    returns = returns + trend

    # Create price series
    close_prices = [base_price]
    for i in range(1, num_bars):
        new_price = close_prices[-1] * (1 + returns[i])
        close_prices.append(max(new_price, 100))

    # Generate OHLC data
    data = []
    for i in range(num_bars):
        close_price = close_prices[i]

        # Generate range based on volatility
        range_pct = np.random.Generator(0.005, 0.06) * (1 + volatility[i] * 5)
        range_value = close_price * range_pct

        # Generate open price
        if i == 0:
            open_price = close_price + np.random.Generator(-range_value / 4, range_value / 4)
        else:
            gap = np.random.Generator(0, 0.002) * close_prices[i - 1]
            open_price = close_prices[i - 1] + gap

        # Generate high and low
        high_price = max(open_price, close_price) + np.random.Generator(0, range_value / 3)
        low_price = min(open_price, close_price) - np.random.Generator(0, range_value / 3)

        # Ensure logical order
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Generate volume
        price_change = abs(close_price - open_price) / open_price
        base_volume = np.random.Generator(500, 2000)
        volume = base_volume * (1 + price_change * 10)

        # Create timestamp
        timestamp = datetime.now() - timedelta(hours=num_bars - i)

        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2)
        })

    return pd.DataFrame(data)


def load_csv_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load OHLCV data from CSV file
    """
    try:
        df = pd.read_csv(file_path)

        # Handle timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])

        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return df

    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None


def save_results(df: pd.DataFrame, cwr_values: pd.Series, analysis: Dict[str, Any]):
    """
    Save results to files
    """
    # Add CWR to dataframe
    df_with_cwr = df.copy()
    df_with_cwr['cwr'] = cwr_values

    # Save to CSV
    df_with_cwr.to_csv('cwr_results.csv', index=False)
    print("✓ Results saved to cwr_results.csv")

    # Save analysis to JSON
    analysis_json = analysis.copy()
    analysis_json['timestamp'] = datetime.now().isoformat()
    analysis_json['data_sample'] = df_with_cwr.head(5).to_dict('records')

    with open('cwr_analysis.json', 'w') as f:
        json.dump(analysis_json, f, indent=2, default=str)
    print("✓ Analysis saved to cwr_analysis.json")


def create_plotly_visualization(df: pd.DataFrame, cwr_values: pd.Series):
    """
    Create interactive visualization using Plotly
    """
    if not HAS_PLOTLY:
        print("Plotly not available. Skipping interactive visualization.")
        return

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Chart with Range Indicators', 'Candle Widening Ratio (CWR)'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )

    # Add range indicators (high-low as bars)
    range_values = df['high'] - df['low']
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=range_values,
            name='Range (High-Low)',
            opacity=0.3,
            yaxis='y2'
        ),
        row=1, col=1
    )

    # Add CWR line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=cwr_values,
            mode='lines',
            name='CWR',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )

    # Add threshold lines to CWR chart
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Baseline (1.0)", row=2, col=1)
    fig.add_hline(y=1.5, line_dash="dash", line_color="red",
                  annotation_text="High Volatility (1.5)", row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="green",
                  annotation_text="Low Volatility (0.5)", row=2, col=1)

    # Color regions based on CWR values
    high_vol_mask = cwr_values > 1.5
    low_vol_mask = cwr_values < 0.5

    if high_vol_mask.any():
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'][high_vol_mask],
                y=cwr_values[high_vol_mask],
                mode='markers',
                name='High Volatility',
                marker=dict(color='red', size=6, symbol='triangle-up')
            ),
            row=2, col=1
        )

    if low_vol_mask.any():
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'][low_vol_mask],
                y=cwr_values[low_vol_mask],
                mode='markers',
                name='Low Volatility',
                marker=dict(color='green', size=6, symbol='triangle-down')
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title="Candle Widening Ratio (CWR) Analysis",
        xaxis_title="Time",
        height=800,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="CWR", row=2, col=1)

    # Save and show
    fig.write_html("cwr_interactive.html")
    print("✓ Interactive visualization saved to cwr_interactive.html")
    fig.show()


def create_matplotlib_visualization(df: pd.DataFrame, cwr_values: pd.Series):
    """
    Create static visualization using Matplotlib
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping static visualization.")
        return

    # Create figure with subplots
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Price chart with range overlay
    ax1.plot(df['timestamp'], df['close'], label='Close Price', color='blue', linewidth=1)
    ax1_twin = ax1.twinx()
    range_values = df['high'] - df['low']
    ax1_twin.bar(df['timestamp'], range_values, alpha=0.3, color='gray', label='Range')
    ax1.set_title('Price Chart with Range Overlay')
    ax1.set_ylabel('Price', color='blue')
    ax1_twin.set_ylabel('Range (High-Low)', color='gray')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. CWR line chart
    ax2.plot(df['timestamp'], cwr_values, label='CWR', color='blue', linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Baseline')
    ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='High Threshold')
    ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Low Threshold')

    # Color regions
    ax2.fill_between(df['timestamp'], 1.5, cwr_values,
                     where=(cwr_values > 1.5), alpha=0.2, color='red', label='High Volatility')
    ax2.fill_between(df['timestamp'], 0.5, cwr_values,
                     where=(cwr_values < 0.5), alpha=0.2, color='green', label='Low Volatility')

    ax2.set_title('Candle Widening Ratio (CWR)')
    ax2.set_ylabel('CWR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. CWR histogram
    ax3.hist(cwr_values.dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(x=1.0, color='gray', linestyle='--', label='Baseline')
    ax3.axvline(x=1.5, color='red', linestyle='--', label='High Threshold')
    ax3.axvline(x=0.5, color='green', linestyle='--', label='Low Threshold')
    ax3.set_title('CWR Distribution')
    ax3.set_xlabel('CWR Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Rolling statistics
    rolling_mean = cwr_values.rolling(window=50).mean()
    rolling_std = cwr_values.rolling(window=50).std()

    ax4.plot(df['timestamp'], rolling_mean, label='50-period Mean', color='blue')
    ax4.fill_between(df['timestamp'],
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     alpha=0.2, color='blue', label='±1 Std Dev')
    ax4.set_title('CWR Rolling Statistics (50-period)')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('CWR')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Format x-axis for better readability
    for ax in [ax1, ax2, ax4]:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('cwr_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Static visualization saved to cwr_analysis.png")
    plt.show()


def print_analysis_summary(analysis: Dict[str, Any]):
    """
    Print a summary of the CWR analysis
    """
    print("\n" + "=" * 50)
    print("CANDLE WIDENING RATIO (CWR) ANALYSIS SUMMARY")
    print("=" * 50)

    stats = analysis['statistics']
    signals = analysis['signals']

    print("\nData Summary:")
    print(f"  Total bars analyzed: {analysis['total_bars']}")
    print(f"  Length parameter: {analysis['length_parameter']}")

    print("\nCWR Statistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Std Dev: {stats['std']:.4f}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print(f"  Skewness: {stats['skewness']:.4f}")
    print(f"  Kurtosis: {stats['kurtosis']:.4f}")

    print("\nVolatility Signals:")
    print(
        f"  High Volatility (>1.5): {signals['high_volatility']['count']} occurrences ({signals['high_volatility']['percentage']:.1f}%)")
    print(
        f"  Low Volatility (<0.5): {signals['low_volatility']['count']} occurrences ({signals['low_volatility']['percentage']:.1f}%)")
    print(
        f"  Normal Volatility: {signals['normal_volatility']['count']} occurrences ({signals['normal_volatility']['percentage']:.1f}%)")

    print("\nRange Behavior:")
    print(
        f"  Expanding Range (>1.0): {signals['expanding_range']['count']} occurrences ({signals['expanding_range']['percentage']:.1f}%)")
    print(
        f"  Contracting Range (<1.0): {signals['contracting_range']['count']} occurrences ({signals['contracting_range']['percentage']:.1f}%)")

    print("\nInterpretation:")
    if stats['mean'] > 1.2:
        print("  • Market shows tendency toward wider ranges (higher volatility)")
    elif stats['mean'] < 0.8:
        print("  • Market shows tendency toward narrower ranges (lower volatility)")
    else:
        print("  • Market shows balanced range behavior")

    if signals['high_volatility']['percentage'] > 10:
        print("  • Frequent high volatility periods detected")
    if signals['low_volatility']['percentage'] > 10:
        print("  • Frequent low volatility periods detected")


def main():
    """
    Main function to run the CWR analysis
    """
    print("Candle Widening Ratio (CWR) Indicator Analysis")
    print("=" * 50)

    # Configuration
    length = 20  # Default length for average calculation

    # Try to load CSV data, fallback to sample data
    csv_file = "ohlcv_data.csv"

    if os.path.exists(csv_file):
        print(f"Loading data from {csv_file}...")
        df = load_csv_data(csv_file)
        if df is None:
            print("Failed to load CSV data. Generating sample data...")
            df = generate_sample_data(500)
    else:
        print(f"CSV file '{csv_file}' not found. Generating sample data...")
        df = generate_sample_data(500)

    print(f"Loaded {len(df)} data points")
    if 'timestamp' in df.columns:
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Initialize CWR calculator
    cwr_calculator = CandleWideningRatio(length=length)

    # Calculate CWR
    print(f"\nCalculating CWR with length={length}...")
    cwr_values = cwr_calculator.calculate(df)

    # Perform analysis
    print("Performing analysis...")
    analysis = cwr_calculator.analyze(df)

    # Print summary
    print_analysis_summary(analysis)

    # Save results
    print("\nSaving results...")
    save_results(df, cwr_values, analysis)

    # Create visualizations
    print("\nCreating visualizations...")
    create_plotly_visualization(df, cwr_values)
    create_matplotlib_visualization(df, cwr_values)

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("Files generated:")
    print("  • cwr_results.csv - Complete data with CWR values")
    print("  • cwr_analysis.json - Detailed analysis results")
    print("  • cwr_interactive.html - Interactive Plotly chart")
    print("  • cwr_analysis.png - Static analysis charts")
    print("\nTo use with your own data:")
    print("  1. Save your OHLCV data as 'ohlcv_data.csv'")
    print("  2. Ensure columns: timestamp, open, high, low, close, volume")
    print("  3. Run this script again")


if __name__ == "__main__":
    main()