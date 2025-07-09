"""
@pyne
"""
from pynecore import Series, Persistent
from pynecore.lib import script, high, low, close, ta, plot, color, input, na


@script.indicator(title="Candle Widening Ratio (CWR)")
def main():
    # Use .value to get actual data
    length = input.int("Length", 20, minval=1, maxval=200).value
    show_thresholds = input.bool("Show Thresholds", True).value
    high_threshold = input.float("High Threshold", 1.5, minval=0.1, step=0.1).value
    low_threshold = input.float("Low Threshold", 0.5, minval=0.1, step=0.1).value

    current_range = high - low
    avg_range = ta.sma(current_range, length)
    cwr = na if avg_range == 0 else current_range / avg_range

    plot(cwr, title="CWR", color=color.blue, linewidth=2)

    if show_thresholds:
        plot(1.0, title="Baseline", color=color.gray, linewidth=1)
        plot(high_threshold, title="High Threshold", color=color.red, linewidth=1)
        plot(low_threshold, title="Low Threshold", color=color.green, linewidth=1)

    high_volatility = cwr > high_threshold
    low_volatility = cwr < low_threshold

    high_vol_count: Persistent[int] = 0
    low_vol_count: Persistent[int] = 0
    total_bars: Persistent[int] = 0

    if not na(cwr):
        total_bars += 1
        if high_volatility:
            high_vol_count += 1
        if low_volatility:
            low_vol_count += 1

    return cwr
