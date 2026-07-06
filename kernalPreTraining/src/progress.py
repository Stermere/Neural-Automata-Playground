"""Live training dashboard: a matplotlib window with a loss-curve plot and a
stats readout, updated in place as training progresses."""
import math
import time

import matplotlib.pyplot as plt

_MAX_EPOCHS = 100000  # drop points older than this many epochs so the chart stays windowed to recent training
_Y_HIGH_PCTL = 90.0  # upper percentile used to clip the y-axis range (ignores rare loss spikes)
_Y_PADDING_FRAC = 0.08  # extra headroom added above/below the clipped range, as a fraction of its span
_Y_MIN_SPAN = 1e-5  # smallest allowed span, to avoid a degenerate/zero-height axis
# (metric key, display label, plot color) - key is what callers pass to update()
_SERIES = (
    ("loss", "loss (total)", "magenta"),
    ("smooth", "loss (smoothed)", "cyan"),
    ("rgb", "rgb loss", "green"),
    ("edge", "edge loss", "goldenrod"),
    ("overflow", "overflow loss", "blue"),
    ("leak", "hidden leak loss", "red"),
)


class TrainingMonitor:
    """Use as a context manager around a training loop:

        with TrainingMonitor(epochs, title="my-run") as monitor:
            for epoch in range(epochs):
                ...
                if epoch % print_every == 0:
                    monitor.update(epoch, loss=..., smooth=..., rgb=..., edge=...,
                                   overflow=..., leak=..., lr=..., best=...)
            monitor.note("done")

    note() prints a status line to the console (checkpoint saves, interrupts)
    without disturbing the chart window.
    """

    def __init__(self, total_epochs, title="training"):
        self.total_epochs = total_epochs
        self.title = title
        self.epochs = []
        self.history = {key: [] for key, _, _ in _SERIES}
        self.start_time = None
        self._epoch = 0
        self._lr = None
        self._best = None

        plt.ion()
        self.fig, (self.ax_chart, self.ax_stats) = plt.subplots(
            1, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [3, 1]},
            constrained_layout=True)
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title(title)
        self.ax_chart.set_xlabel("epoch")
        self.ax_chart.set_ylabel("value")
        self.ax_chart.grid(True, alpha=0.3)
        self._lines = {
            key: self.ax_chart.plot([], [], label=label, color=color, linewidth=1.2)[0]
            for key, label, color in _SERIES
        }
        self.ax_stats.axis("off")
        self._stats_text = self.ax_stats.text(
            0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=10,
            transform=self.ax_stats.transAxes)

    def __enter__(self):
        self.start_time = time.time()
        plt.show(block=False)
        return self

    def __exit__(self, *exc_info):
        plt.ioff()
        return False

    def note(self, message):
        print(message)

    def update(self, epoch, lr=None, best=None, **metrics):
        self._epoch = epoch
        self._lr = lr
        self._best = best
        self.epochs.append(epoch)
        for key, _, _ in _SERIES:
            self.history[key].append(metrics.get(key))
        cutoff = epoch - _MAX_EPOCHS
        drop = 0
        while drop < len(self.epochs) and self.epochs[drop] <= cutoff:
            drop += 1
        if drop:
            self.epochs = self.epochs[drop:]
            for name in self.history:
                self.history[name] = self.history[name][drop:]

        combined_ys = []
        for key, _, _ in _SERIES:
            values = self.history[key]
            xs = [e for e, v in zip(self.epochs, values) if v is not None]
            ys = [v for v in values if v is not None]
            self._lines[key].set_data(xs, ys)
            self._lines[key].set_visible(bool(ys))
            combined_ys.extend(ys)

        active = [line for line in self._lines.values() if line.get_visible()]
        if active:
            self.ax_chart.legend(handles=active, loc="upper right", fontsize=8)
        self.ax_chart.relim()
        self.ax_chart.autoscale_view(scaley=False)
        self._apply_y_limits(combined_ys)
        self.ax_chart.set_title(f"{self.title} — epoch {epoch}/{self.total_epochs}")


        self._stats_text.set_text(self._stats_str())

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _apply_y_limits(self, values):
        """Clip the y-axis to a robust range so rare loss spikes don't
        squash the rest of the chart flat."""
        finite = [v for v in values if v is not None and math.isfinite(v)]
        if not finite:
            return
        low = min(finite)
        high = _percentile(finite, _Y_HIGH_PCTL)
        if high < low:
            high = low
        span = high - low
        if span < _Y_MIN_SPAN:
            mid = (high + low) / 2
            low = mid - _Y_MIN_SPAN / 2
            high = mid + _Y_MIN_SPAN / 2
            span = high - low
        pad = span * _Y_PADDING_FRAC
        self.ax_chart.set_ylim(low - pad, high + pad)

    def _stats_str(self):
        lines = [f"{'epoch':<18} {self._epoch}/{self.total_epochs}"]
        for key, label, _ in _SERIES:
            values = self.history[key]
            if values and values[-1] is not None:
                lines.append(f"{label:<18} {values[-1]:.5f}")
        if self._best is not None:
            lines.append(f"{'best (eval)':<18} {self._best:.5f}")
        if self._lr is not None:
            lines.append(f"{'lr':<18} {self._lr:.2e}")
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            rate = self._epoch / elapsed if elapsed > 0 and self._epoch > 0 else 0
            remaining = self.total_epochs - self._epoch
            eta = remaining / rate if rate > 0 else None
            lines.append(f"{'elapsed':<18} {_format_duration(elapsed)}")
            if eta is not None:
                lines.append(f"{'eta':<18} {_format_duration(eta)}")
        return "\n".join(lines)


def _percentile(values, pct):
    """Linear-interpolated percentile over a list of numbers (no numpy dependency)."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    rank = (pct / 100.0) * (n - 1)
    low_idx = int(math.floor(rank))
    high_idx = int(math.ceil(rank))
    if low_idx == high_idx:
        return ordered[low_idx]
    frac = rank - low_idx
    return ordered[low_idx] + (ordered[high_idx] - ordered[low_idx]) * frac


def _format_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
