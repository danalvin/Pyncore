"""
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
