import numpy as np


def stance_to_step_times(segments):
    return np.array([(s + e) / 2 for s, e, _ in segments])


def same_foot_step_times(step_times):
    return np.diff(step_times)


def alternating_step_times(tL, tR):
    events = [(t, "L") for t in tL] + [(t, "R") for t in tR]
    events.sort()

    steps = []
    for (t0, f0), (t1, f1) in zip(events, events[1:]):
        if f0 != f1:
            steps.append(t1 - t0)

    return np.array(steps)


def cadence(step_times):
    if len(step_times) == 0:
        return None
    return 60.0 / np.mean(step_times)


def symmetry_index(a, b):
    if len(a) == 0 or len(b) == 0:
        return None
    return 100 * abs(np.mean(a) - np.mean(b)) / ((np.mean(a) + np.mean(b)) / 2)
