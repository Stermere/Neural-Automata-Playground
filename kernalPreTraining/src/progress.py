"""Live training dashboard: a plotext loss curve plus a stats table, redrawn
in place with rich.Live instead of scrolling the terminal with one print per
logging step."""
import sys
import time

import plotext as plt
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# The chart uses Unicode braille/box characters; when stdout is redirected or
# the console is legacy Windows, its encoding can be cp1252 and printing the
# dashboard raises UnicodeEncodeError. Degrade to '?' instead of crashing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_MAX_POINTS = 400  # halve resolution past this so long runs stay responsive
# (metric key, display label, plot color) - key is what callers pass to update()
_SERIES = (
    ("loss", "loss (total)", "magenta"),
    ("smooth", "loss (smoothed)", "cyan"),
    ("rgb", "rgb MSE", "green"),
    ("edge", "edge MSE", "yellow"),
    ("overflow", "overflow", "blue"),
    ("leak", "hidden leak MSE", "red"),
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

    note() prints a line above the live region (for one-off messages like
    checkpoint saves) instead of disrupting the chart.
    """

    def __init__(self, total_epochs, title="training", console=None):
        self.total_epochs = total_epochs
        self.title = title
        self.console = console or Console()
        self.epochs = []
        self.history = {key: [] for key, _, _ in _SERIES}
        self.start_time = None
        self._epoch = 0
        self._lr = None
        self._best = None
        self._live = Live(self._render(), console=self.console,
                          refresh_per_second=4, transient=False)

    def __enter__(self):
        self.start_time = time.time()
        self._live.__enter__()
        return self

    def __exit__(self, *exc_info):
        self._live.__exit__(*exc_info)

    def note(self, message):
        self._live.console.print(message)

    def update(self, epoch, lr=None, best=None, **metrics):
        self._epoch = epoch
        self._lr = lr
        self._best = best
        self.epochs.append(epoch)
        for key, _, _ in _SERIES:
            self.history[key].append(metrics.get(key))
        if len(self.epochs) > _MAX_POINTS:
            self.epochs = self.epochs[::2]
            for name in self.history:
                self.history[name] = self.history[name][::2]
        self._live.update(self._render(), refresh=True)

    def _chart(self):
        if len(self.epochs) < 2:
            return Text("(warming up...)")
        plot_width = max(40, self.console.width - 4)
        plt.clf()
        plt.plotsize(plot_width, 18)
        plt.theme("pro")
        plt.xlabel("epoch")
        for key, label, color in _SERIES:
            values = self.history[key]
            if any(v is not None for v in values):
                xs = [e for e, v in zip(self.epochs, values) if v is not None]
                ys = [v for v in values if v is not None]
                plt.plot(xs, ys, label=label, color=color)
        return Text.from_ansi(plt.build())

    def _stats(self):
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold")
        table.add_column()
        table.add_row("epoch", f"{self._epoch}/{self.total_epochs}")
        for key, label, _ in _SERIES:
            values = self.history[key]
            if values and values[-1] is not None:
                table.add_row(label, f"{values[-1]:.5f}")
        if self._best is not None:
            table.add_row("best (smoothed)", f"{self._best:.5f}")
        if self._lr is not None:
            table.add_row("lr", f"{self._lr:.2e}")
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            rate = self._epoch / elapsed if elapsed > 0 and self._epoch > 0 else 0
            remaining = self.total_epochs - self._epoch
            eta = remaining / rate if rate > 0 else None
            table.add_row("elapsed", _format_duration(elapsed))
            if eta is not None:
                table.add_row("eta", _format_duration(eta))
        return table

    def _render(self):
        return Panel(Group(self._chart(), self._stats()), title=self.title)


def _format_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
