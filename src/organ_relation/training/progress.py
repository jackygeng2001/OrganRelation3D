"""Independent rolling clocks; no tensors, no speed state persisted across resume."""
from collections import deque
from datetime import datetime, timedelta, timezone
import math


class ProgressClock:
    def __init__(self, window=20, warmup=5):
        if type(window) is not int or type(warmup) is not int or not 1 <= warmup <= window:
            raise ValueError('require 1 <= warmup <= window')
        self.samples = deque(maxlen=window)
        self.warmup = warmup

    def add(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('elapsed seconds must be finite and nonnegative')
        self.samples.append(float(seconds))

    def estimate(self, remaining, epoch_remaining=None):
        if remaining < 0 or (epoch_remaining is not None and not 0 <= epoch_remaining <= remaining):
            raise ValueError('invalid remaining work')
        ready = len(self.samples) >= self.warmup
        mean = sum(self.samples) / len(self.samples) if self.samples else None
        eta = 0.0 if remaining == 0 else mean * remaining if ready else None
        epoch_eta = (0.0 if epoch_remaining == 0 else mean * epoch_remaining if ready else None
                     ) if epoch_remaining is not None else None
        return dict(status='complete' if remaining == 0 else 'estimating' if ready else 'warming up',
                    last_seconds=self.samples[-1] if self.samples else None, rolling_seconds=mean,
                    eta_seconds=eta, epoch_eta_seconds=epoch_eta,
                    estimated_finish_utc=(datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat()
                    if eta is not None else None)
