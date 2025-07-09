# PyneCore Direct Usage - CWR Indicator Implementation

## Overview

This project demonstrates direct usage of PyneCore through Python to implement a Candle Widening Ratio (CWR) indicator with comprehensive visualization and analysis capabilities.

## What is CWR?

The Candle Widening Ratio (CWR) is a volatility indicator that measures how the current candle's range compares to the average range of previous candles:

```
CWR = (current_range) / (average_range_of_last_N_candles)
where current_range = high - low
```

### Interpretation:

* **CWR > 1.0**: Current candle has wider range than average (higher volatility)
* **CWR < 1.0**: Current candle has narrower range than average (lower volatility)
* **CWR = 1.0**: Current candle matches average range

## Features

### Core Functionality

* ✅ PyneCore Script: Complete CWR indicator implementation
* ✅ Sample Data Generation: Realistic OHLCV data simulation
* ✅ Direct Python Execution: No CLI required
* ✅ Multiple Output Formats: JSON, CSV results
* ✅ Statistical Analysis: Comprehensive volatility metrics

### Visualizations

* 📊 Interactive Price Chart: Candlestick chart with volume overlay
* 📈 CWR Indicator: Real-time indicator with threshold lines
* 🎯 Volatility Markers: Visual highlights for high/low volatility periods
* 📋 Statistics Dashboard: Distribution analysis and metrics
* 🖱️ Interactive Features: Hover tooltips and zoom controls

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python direct_use/pynecore_direct_usage.py
```

## Generated Files

### Core Files

* `cwr_indicator.py` - PyneCore script implementation
* `sample_data.csv` - Generated OHLCV data
* `cwr_pynecore_results.json` - Complete analysis results
* `cwr_pynecore_data.csv` - Data with calculated CWR values
* `cwr_pynecore_summary.json` - Summary statistics

### Visualizations

* `cwr_analysis_chart.html` - Interactive price & CWR chart
* `cwr_statistics_dashboard.html` - CWR statistics dashboard

## CWR Indicator Parameters

| Parameter         | Default | Description                                   |
| ----------------- | ------- | --------------------------------------------- |
| `length`          | 20      | Lookback period for average range calculation |
| `high_threshold`  | 1.5     | Threshold for high volatility detection       |
| `low_threshold`   | 0.5     | Threshold for low volatility detection        |
| `show_thresholds` | True    | Display threshold lines on chart              |

## Statistical Metrics

The implementation provides comprehensive statistics:

* Mean/Std/Min/Max CWR values
* High volatility periods (CWR > 1.5)
* Low volatility periods (CWR < 0.5)
* Distribution analysis
* Recent values tracking

## Architecture

### PyneCore Script Structure

```python
@script.indicator(title="Candle Widening Ratio (CWR)")
def main(length: int = 20, ...):
    current_range = high - low
    avg_range = ta.sma(current_range, length)
    cwr = current_range / avg_range
    plot(cwr, title="CWR", color=color.blue)
    return cwr
```

### Direct Python Execution

The implementation simulates PyneCore's internal execution:

* **Data Loading**: CSV → DataFrame
* **Indicator Calculation**: Rolling average range computation
* **Statistical Analysis**: Volatility period detection
* **Result Generation**: JSON/CSV outputs
* **Visualization**: Interactive Plotly charts

## Sample Output

**CWR Statistics:**

```
Mean: 1.0234
Std: 0.4567
Min: 0.1234
Max: 3.4567
High Volatility: 45 (9.0%)
Low Volatility: 67 (13.4%)
```

## Visualization Features

### Main Chart

* Candlestick Chart: OHLC price action with volume
* CWR Indicator: Real-time volatility measurement
* Threshold Lines: Configurable high/low volatility levels
* Volatility Markers: Visual indicators for extreme periods

### Statistics Dashboard

* CWR Distribution: Histogram of indicator values
* Volatility Breakdown: Pie chart of volatility periods
* Statistical Summary: Key metrics visualization
* Recent Values: Time series of latest readings

## Technical Implementation

### Data Generation

Realistic market data simulation with:

* **Trend Components**: Sine wave overlay
* **Volatility Clustering**: Time-varying volatility
* **Logical OHLC**: Proper high/low/open/close relationships
* **Volume Simulation**: Realistic trading volume

### Performance Optimizations

* Efficient Rolling Calculations: Pandas rolling operations
* Memory Management: Streaming data processing
* Vectorized Operations: NumPy-based computations

## Future Enhancements

* ⏰ Real-time Data: WebSocket integration
* 🔢 Multiple Timeframes: Multi-timeframe analysis
* 🚨 Alert System: Volatility breakout notifications
* ⏲️ Backtesting: Historical performance analysis
* 📄 Export Options: PNG/PDF chart exports

## License

This project demonstrates PyneCore usage patterns and is provided as-is for educational purposes.
