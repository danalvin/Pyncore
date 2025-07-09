#!/usr/bin/env python3
"""
PyneCore Direct Usage Example - CWR Indicator with Visualization
This demonstrates how to use PyneCore directly from Python with interactive charts
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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
    volatility = np.random.uniform(0.01, 0.03, num_bars)
    returns = np.random.normal(0, 1, num_bars) * volatility

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
        range_pct = np.random.uniform(0.005, 0.04) * (1 + volatility[i] * 3)
        range_value = close_price * range_pct

        # Generate OHLC
        if i == 0:
            open_price = close_price + np.random.uniform(-range_value / 4, range_value / 4)
        else:
            gap = np.random.normal(0, 0.001) * close_prices[i - 1]
            open_price = close_prices[i - 1] + gap

        high_price = max(open_price, close_price) + np.random.uniform(0, range_value / 2)
        low_price = min(open_price, close_price) - np.random.uniform(0, range_value / 2)

        # Ensure logical order
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Generate volume
        volume = np.random.uniform(1000, 5000)

        # Create timestamp (assuming 1-hour bars)
        timestamp = datetime.now() - timedelta(hours=num_bars - i)

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


def simulate_pynecore_execution(script_path: str, data_path: str):
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
    df['current_range'] = current_range
    df['avg_range'] = avg_range

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

    print(f"\nCWR Statistics:")
    print(f"  Mean: {results['statistics']['mean_cwr']:.4f}")
    print(f"  Std: {results['statistics']['std_cwr']:.4f}")
    print(f"  Min: {results['statistics']['min_cwr']:.4f}")
    print(f"  Max: {results['statistics']['max_cwr']:.4f}")
    print(
        f"  High Volatility: {results['statistics']['high_volatility_count']} ({results['statistics']['high_volatility_percentage']:.1f}%)")
    print(
        f"  Low Volatility: {results['statistics']['low_volatility_count']} ({results['statistics']['low_volatility_percentage']:.1f}%)")

    return results


def create_interactive_visualization(results: dict):
    """Create interactive visualization using Plotly"""

    df = pd.DataFrame(results['data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Get parameters
    params = results['parameters']
    high_threshold = params['high_threshold']
    low_threshold = params['low_threshold']

    # Debug: Print data info
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    print(f"   CWR range: {df['cwr'].min():.3f} to {df['cwr'].max():.3f}")

    # Create subplots: price chart on top, CWR indicator below
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price Chart with Volume', 'Candle Widening Ratio (CWR)'),
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # ===== TOP SUBPLOT: CANDLESTICK CHART =====
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='rgba(0, 255, 136, 0.3)',
            decreasing_fillcolor='rgba(255, 68, 68, 0.3)'
        ),
        row=1, col=1
    )

    # Add volume bars on secondary y-axis
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color='rgba(158, 158, 158, 0.3)',
            showlegend=True
        ),
        row=1, col=1, secondary_y=True
    )

    # ===== BOTTOM SUBPLOT: CWR INDICATOR =====

    # Main CWR line
    cwr_trace = go.Scatter(
        x=df['timestamp'],
        y=df['cwr'],
        mode='lines',
        name='CWR',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='<b>CWR</b>: %{y:.3f}<br><b>Date</b>: %{x}<extra></extra>'
    )

    fig.add_trace(cwr_trace, row=2, col=1)

    # Add threshold lines
    baseline = go.Scatter(
        x=df['timestamp'],
        y=[1.0] * len(df),
        mode='lines',
        name='Baseline (1.0)',
        line=dict(color='gray', width=1, dash='dash'),
        hovertemplate='<b>Baseline</b>: 1.0<extra></extra>'
    )

    fig.add_trace(baseline, row=2, col=1)

    high_threshold_line = go.Scatter(
        x=df['timestamp'],
        y=[high_threshold] * len(df),
        mode='lines',
        name=f'High Threshold ({high_threshold})',
        line=dict(color='red', width=1, dash='dot'),
        hovertemplate=f'<b>High Threshold</b>: {high_threshold}<extra></extra>'
    )

    fig.add_trace(high_threshold_line, row=2, col=1)

    low_threshold_line = go.Scatter(
        x=df['timestamp'],
        y=[low_threshold] * len(df),
        mode='lines',
        name=f'Low Threshold ({low_threshold})',
        line=dict(color='green', width=1, dash='dot'),
        hovertemplate=f'<b>Low Threshold</b>: {low_threshold}<extra></extra>'
    )

    fig.add_trace(low_threshold_line, row=2, col=1)

    # Highlight high/low volatility areas
    high_vol_mask = df['cwr'] > high_threshold
    low_vol_mask = df['cwr'] < low_threshold

    if high_vol_mask.any():
        high_vol_scatter = go.Scatter(
            x=df[high_vol_mask]['timestamp'],
            y=df[high_vol_mask]['cwr'],
            mode='markers',
            name='High Volatility',
            marker=dict(color='red', size=6, symbol='triangle-up'),
            hovertemplate='<b>High Volatility</b><br>CWR: %{y:.3f}<br>Date: %{x}<extra></extra>'
        )
        fig.add_trace(high_vol_scatter, row=2, col=1)

    if low_vol_mask.any():
        low_vol_scatter = go.Scatter(
            x=df[low_vol_mask]['timestamp'],
            y=df[low_vol_mask]['cwr'],
            mode='markers',
            name='Low Volatility',
            marker=dict(color='green', size=6, symbol='triangle-down'),
            hovertemplate='<b>Low Volatility</b><br>CWR: %{y:.3f}<br>Date: %{x}<extra></extra>'
        )
        fig.add_trace(low_vol_scatter, row=2, col=1)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Candle Widening Ratio (CWR) Analysis<br><sub>Length: {params['length']}, High Threshold: {high_threshold}, Low Threshold: {low_threshold}</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, df['volume'].max() * 4]  # Scale volume to background
        ),

        template="plotly_dark",
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

    # Update y-axis titles for each subplot
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="CWR Value", row=2, col=1)

    # Remove rangeslider (it's too small for this use case)
    fig.update_layout(xaxis_rangeslider_visible=False)

    return fig


def create_statistics_visualization(results: dict):
    """Create additional statistics visualization"""

    stats = results['statistics']

    # Create a summary dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'CWR Distribution',
            'Volatility Breakdown',
            'CWR Statistics',
            'Recent CWR Values'
        ),
        specs=[[{"type": "histogram"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )

    df = pd.DataFrame(results['data'])

    # 1. CWR Distribution (Histogram)
    fig.add_trace(
        go.Histogram(
            x=df['cwr'],
            nbinsx=30,
            name='CWR Distribution',
            marker_color='#2E86AB',
            opacity=0.7
        ),
        row=1, col=1
    )

    # 2. Volatility Breakdown (Pie Chart)
    normal_vol = stats['total_bars'] - stats['high_volatility_count'] - stats['low_volatility_count']

    fig.add_trace(
        go.Pie(
            labels=['Normal Volatility', 'High Volatility', 'Low Volatility'],
            values=[normal_vol, stats['high_volatility_count'], stats['low_volatility_count']],
            marker_colors=['#A23B72', '#F18F01', '#C73E1D'],
            hole=0.3
        ),
        row=1, col=2
    )

    # 3. CWR Statistics (Bar Chart)
    stat_names = ['Mean', 'Std', 'Min', 'Max']
    stat_values = [stats['mean_cwr'], stats['std_cwr'], stats['min_cwr'], stats['max_cwr']]

    fig.add_trace(
        go.Bar(
            x=stat_names,
            y=stat_values,
            name='CWR Stats',
            marker_color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        ),
        row=2, col=1
    )

    # 4. Recent CWR Values (Line Chart)
    recent_values = results['recent_values']
    recent_x = list(range(len(recent_values)))

    fig.add_trace(
        go.Scatter(
            x=recent_x,
            y=recent_values,
            mode='lines+markers',
            name='Recent CWR',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6)
        ),
        row=2, col=2
    )

    fig.update_layout(
        title="CWR Analysis Dashboard",
        template="plotly_dark",
        height=600,
        showlegend=False
    )

    return fig


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
    Main function demonstrating direct PyneCore usage with visualization
    """

    print("PyneCore Direct Usage - CWR Indicator with Visualization")
    print("=" * 60)

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
    results = simulate_pynecore_execution(script_path, data_path)

    if results:
        # Step 4: Save results
        print("\n4. Saving results...")
        save_results_to_files(results)

        # Step 5: Create visualizations
        print("\n5. Creating visualizations...")

        # Main chart with price and CWR
        main_chart = create_interactive_visualization(results)
        main_chart.write_html("cwr_analysis_chart.html")
        print("   ✓ Interactive chart saved to cwr_analysis_chart.html")

        # Statistics dashboard
        stats_chart = create_statistics_visualization(results)
        stats_chart.write_html("cwr_statistics_dashboard.html")
        print("   ✓ Statistics dashboard saved to cwr_statistics_dashboard.html")

        # Show the charts (optional - will open in browser)
        try:
            print("\n6. Opening charts in browser...")
            main_chart.show()
            stats_chart.show()
        except Exception as e:
            print(f"   Note: Could not open charts in browser: {e}")
            print("   You can manually open the .html files to view the charts")

        print("\n" + "=" * 60)
        print("EXECUTION COMPLETE")
        print("=" * 60)
        print("Files generated:")
        print("  • cwr_indicator.py - PyneCore script")
        print("  • sample_data.csv - Sample OHLCV data")
        print("  • cwr_pynecore_results.json - Complete results")
        print("  • cwr_pynecore_data.csv - Data with CWR values")
        print("  • cwr_pynecore_summary.json - Summary statistics")
        print("  • cwr_analysis_chart.html - Interactive price & CWR chart")
        print("  • cwr_statistics_dashboard.html - CWR statistics dashboard")

        print(f"\nVisualization Features:")
        print(f"  • Interactive candlestick chart with volume")
        print(f"  • CWR indicator with threshold lines")
        print(f"  • High/low volatility markers")
        print(f"  • Statistics dashboard with distribution analysis")
        print(f"  • Hover tooltips for detailed information")

        print(f"\nRequired dependencies:")
        print(f"  pip install plotly pandas numpy")

    else:
        print("Execution failed!")


if __name__ == "__main__":
    main()