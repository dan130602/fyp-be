"""Original energy boundaries, retained unchanged from the audited experiment."""
import numpy as np
RATE=16000
MAX_SAMPLES=30*RATE

def chunk_bounds(waveform, pause_aware, *, midpoint=False):
    """Move a cut back to a quiet 200 ms window when one is available."""
    start = 0
    frame = RATE // 50
    while start < len(waveform):
        end = min(start + MAX_SAMPLES, len(waveform))
        if pause_aware and end < len(waveform):
            search = waveform[end - 4 * RATE:end]
            power = np.mean(search.reshape(-1, frame) ** 2, axis=1)
            rms = np.sqrt(np.convolve(power, np.ones(10) / 10, mode="valid"))
            quiet = max(0.001, min(0.01, 0.35 * float(np.median(np.sqrt(power)))))
            choices = np.flatnonzero(rms <= quiet)
            if len(choices):
                window = int(choices[-1]) + 5
                if midpoint:
                    stretches = np.split(choices, np.flatnonzero(np.diff(choices) > 1) + 1)
                    longest = max(stretches, key=lambda stretch: (len(stretch), stretch[-1]))
                    # Each qualifying window covers ten frames. Take the midpoint
                    # of the union of consecutive windows.
                    window = (int(longest[0]) + int(longest[-1]) + 10) // 2
                end = end - 4 * RATE + window * frame
        yield start, end
        start = end
