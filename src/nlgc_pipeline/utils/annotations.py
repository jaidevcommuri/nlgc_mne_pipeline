from __future__ import annotations
import mne


def trial_relative_annotations(
    annotations: mne.Annotations,
    *,
    trial_start_s: float,
    trial_stop_s: float,
) -> mne.Annotations:
    """
    Clip annotations to one trial and express their onsets relative to it.

    The returned annotations have `orig_time=None`, so onset=0 means the
    beginning of the saved trial/evoked data regardless of the original
    session-level annotation clock.
    """
    onsets = []
    durations = []
    descriptions = []

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):
        annotation_start_s = float(onset)
        annotation_stop_s = annotation_start_s + float(duration)

        overlap_start_s = max(annotation_start_s, trial_start_s)
        overlap_stop_s = min(annotation_stop_s, trial_stop_s)

        if overlap_stop_s <= overlap_start_s:
            continue

        onsets.append(overlap_start_s - trial_start_s)
        durations.append(overlap_stop_s - overlap_start_s)
        descriptions.append(str(description))

    return mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=None,
    )