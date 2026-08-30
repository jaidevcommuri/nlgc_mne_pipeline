import numpy as np
import mne
import pandas as pd


def robust_noisy_meg_channels(
    raw,
    picks=None,
    window_s=5.0,
    overlap=0.5,
    z_threshold=5.0,
    bad_fraction_threshold=0.20,
    ptp_z_threshold=5.0,
):
    """
    Flag persistently high-amplitude MEG channels using robust windowed
    variance and peak-to-peak outlier scores.

    Evaluate mag and grad channels separately. Returned channels should be
    visually checked before being committed to raw.info['bads'].
    """
    if picks is None:
        picks = mne.pick_types(
            raw.info,
            meg=True,
            eeg=False,
            eog=False,
            ecg=False,
            stim=False,
            exclude=[],
        )

    sfreq = raw.info["sfreq"]
    n_win = max(2, int(round(window_s * sfreq)))
    hop = max(1, int(round(n_win * (1.0 - overlap))))

    data = raw.get_data(picks=picks, reject_by_annotation="omit")
    ch_names = np.array([raw.ch_names[p] for p in picks])

    starts = np.arange(0, data.shape[1] - n_win + 1, hop)
    if starts.size == 0:
        raise ValueError("Recording is shorter than one analysis window.")

    log_var = np.empty((len(starts), len(picks)))
    log_ptp = np.empty((len(starts), len(picks)))

    for i, start in enumerate(starts):
        segment = data[:, start:start + n_win]
        var = np.var(segment, axis=1)
        ptp = np.ptp(segment, axis=1)

        log_var[i] = np.log(np.maximum(var, np.finfo(float).tiny))
        log_ptp[i] = np.log(np.maximum(ptp, np.finfo(float).tiny))

    def robust_z(x):
        med = np.median(x, axis=1, keepdims=True)
        mad = np.median(np.abs(x - med), axis=1, keepdims=True)
        scale = 1.4826 * np.maximum(mad, np.finfo(float).eps)
        return (x - med) / scale

    z_var = robust_z(log_var)
    z_ptp = robust_z(log_ptp)

    outlier_windows = (
        (z_var > z_threshold)
        | (z_ptp > ptp_z_threshold)
    )

    outlier_fraction = outlier_windows.mean(axis=0)
    median_z_var = np.median(z_var, axis=0)
    median_z_ptp = np.median(z_ptp, axis=0)

    flagged = outlier_fraction >= bad_fraction_threshold

    scores = {
        name: {
            "outlier_window_fraction": float(outlier_fraction[i]),
            "median_logvar_z": float(median_z_var[i]),
            "median_logptp_z": float(median_z_ptp[i]),
            "flagged": bool(flagged[i]),
        }
        for i, name in enumerate(ch_names)
    }

    return ch_names[flagged].tolist(), scores


def maxwell_flat_qc(
    auto_scores,
    auto_flat_chs=None,
    global_flat_fraction=0.20,
    persistent_flat_fraction=0.80,
    min_persistent_bins=3,
    min_global_bins=1,
    pad_sec=0.0,
    annotation_label="BAD_GLOBAL_ARTIFACT",
):
    """
    Separate transient/global flatness events from persistently flat MEG channels.

    Parameters
    ----------
    auto_scores : dict
        Output from mne.preprocessing.find_bad_channels_maxwell(...,
        return_scores=True). Required keys: 'ch_names', 'bins',
        'scores_flat', 'limits_flat'.
    auto_flat_chs : sequence of str | None
        The returned auto-flat channel list. Used only to infer/check whether
        MNE's criterion is score < limit or score > limit.
    global_flat_fraction : float
        Fraction of scored channels flat in a bin required to call the entire
        bin a BAD segment. Start around 0.20–0.35 for broad clipping.
    persistent_flat_fraction : float
        Fraction of non-global bins in which a channel must be flat to be
        designated a persistent/global bad channel.
    min_persistent_bins : int
        Require at least this many flat non-global bins for a channel-level call.
    min_global_bins : int
        Require this many consecutive globally-flat bins before annotating.
        With 5-second Maxwell bins, 1 detects events >= 5 sec.
    pad_sec : float
        Extend each returned bad annotation on both sides.
    annotation_label : str
        Must begin with 'BAD' if it will be used with
        reject_by_annotation / skip_by_annotation.

    Returns
    -------
    persistent_flat_chs : list of str
        Channels that are flat in a high fraction of otherwise non-global bins.
    annotations : mne.Annotations
        BAD annotations for merged global-flat bins.
    qc : dict
        Full bin/channel diagnostics for saving as QC output.
    """
    required = {"ch_names", "bins", "scores_flat", "limits_flat"}
    missing = required.difference(auto_scores)
    if missing:
        raise KeyError(f"auto_scores is missing keys: {sorted(missing)}")

    ch_names = np.asarray(auto_scores["ch_names"], dtype=str)
    bins = np.asarray(auto_scores["bins"], dtype=float)
    scores = np.asarray(auto_scores["scores_flat"], dtype=float)
    limits = np.asarray(auto_scores["limits_flat"].flatten(), dtype=float)

    if bins.ndim != 2 or bins.shape[1] != 2:
        raise ValueError(
            f"`bins` must have shape (n_bins, 2), got {bins.shape}."
        )

    n_ch = len(ch_names)
    n_bins = len(bins)

    if scores.shape == (n_ch, n_bins):
        pass
    elif scores.shape == (n_bins, n_ch):
        scores = scores.T
    else:
        raise ValueError(
            "`scores_flat` must have shape (n_channels, n_bins) or "
            f"(n_bins, n_channels); got {scores.shape}, expected "
            f"({n_ch}, {n_bins}) or ({n_bins}, {n_ch})."
        )

    # Broadcast limits to shape (n_channels, n_bins).
    if limits.ndim == 0:
        limits_2d = np.full_like(scores, float(limits))
    elif limits.shape == (n_ch,):
        limits_2d = np.broadcast_to(limits[:, np.newaxis], scores.shape)
    elif limits.shape == (n_bins,):
        limits_2d = np.broadcast_to(limits[np.newaxis, :], scores.shape)
    elif limits.shape == scores.shape:
        limits_2d = limits
    elif limits.shape == scores.T.shape:
        limits_2d = limits.T
    else:
        raise ValueError(
            f"Cannot align limits_flat shape {limits.shape} to scores shape "
            f"{scores.shape}."
        )

    # MNE flatness is normally "score < limit": low variation is flat.
    flat_lt = scores < limits_2d
    flat_gt = scores > limits_2d

    criterion = "<"
    flat_mask = flat_lt

    # Validate direction from MNE's collapsed auto-flat decision if provided.
    if auto_flat_chs is not None and len(auto_flat_chs):
        auto_set = set(auto_flat_chs)
        observed_auto = np.array([name in auto_set for name in ch_names])

        # A channel is called at least once if it is flagged in any bin.
        pred_lt = flat_lt.any(axis=1)
        pred_gt = flat_gt.any(axis=1)

        agreement_lt = np.mean(pred_lt == observed_auto)
        agreement_gt = np.mean(pred_gt == observed_auto)

        if agreement_gt > agreement_lt:
            criterion = ">"
            flat_mask = flat_gt
        else:
            criterion = "<"
            flat_mask = flat_lt

    # Fraction of channels flat in every Maxwell time bin.
    bin_flat_fraction = flat_mask.mean(axis=0)

    global_bin_mask = bin_flat_fraction >= global_flat_fraction

    # Require a minimum consecutive duration expressed in bins.
    if min_global_bins > 1:
        kernel = np.ones(min_global_bins, dtype=int)
        run_counts = np.convolve(
            global_bin_mask.astype(int), kernel, mode="same"
        )
        global_bin_mask = run_counts >= min_global_bins

    # Persistent flat-channel estimate excludes broad/global bad intervals.
    usable_bins = ~global_bin_mask

    if usable_bins.sum() == 0:
        raise RuntimeError(
            "All Maxwell bins were classified as globally bad. Relax "
            "`global_flat_fraction` or inspect this recording manually."
        )

    channel_flat_count = flat_mask[:, usable_bins].sum(axis=1)
    channel_flat_fraction = flat_mask[:, usable_bins].mean(axis=1)

    persistent_channel_mask = (
        (channel_flat_fraction >= persistent_flat_fraction)
        & (channel_flat_count >= min_persistent_bins)
    )

    persistent_flat_chs = ch_names[persistent_channel_mask].tolist()

    # Merge consecutive global bins into time intervals.
    transitions = np.flatnonzero(
        np.diff(np.r_[False, global_bin_mask, False].astype(int))
    )

    onset = []
    duration = []

    for start_bin, stop_bin in transitions.reshape(-1, 2):
        start = bins[start_bin, 0]
        stop = bins[stop_bin - 1, 1]

        start = max(0.0, start - pad_sec)
        stop = stop + pad_sec

        onset.append(start)
        duration.append(stop - start)

    annotations = mne.Annotations(
        onset=onset,
        duration=duration,
        description=[annotation_label] * len(onset),
    )

    qc = {
        "ch_names": ch_names,
        "bins": bins,
        "scores_flat": scores,
        "limits_flat": limits_2d,
        "criterion": criterion,
        "flat_mask": flat_mask,
        "bin_flat_fraction": bin_flat_fraction,
        "global_bin_mask": global_bin_mask,
        "usable_bin_mask": usable_bins,
        "channel_flat_count": channel_flat_count,
        "channel_flat_fraction": channel_flat_fraction,
        "persistent_channel_mask": persistent_channel_mask,
        "global_flat_fraction_threshold": global_flat_fraction,
        "persistent_flat_fraction_threshold": persistent_flat_fraction,
    }

    return persistent_flat_chs, annotations, qc


def score_ica_cardiac(
    raw,
    ica,
    ecg_events=None,
    tmin=-0.20,
    tmax=0.50,
    baseline=(-0.20, -0.05),
    min_events=12,
    min_bpm=35.0,
    max_bpm=180.0,
    reject_threshold=0.20,
    min_r2_lock=0.02,
    min_split_r=0.50,
):
    """
    Deterministic MEG-only cardiac ICA scoring with no permutation/null loop.

    Uses synthetic R events when `ecg_events` is None. Returns one bounded
    score per ICA component and a binary `reject` decision.

    The score combines:
      1. R-locked variance fraction,
      2. odd/even R-locked waveform reproducibility,
      3. sign-invariant similarity to a robust cardiac template.
    """
    sfreq = float(raw.info["sfreq"])

    if ecg_events is None:
        ecg_events, _, _, _ = mne.preprocessing.find_ecg_events(
            raw,
            ch_name=None,
            event_id=999,
            l_freq=8.0,
            h_freq=16.0,
            reject_by_annotation=True,
            return_ecg=True,
            verbose=False,
        )

    event_samples = np.asarray(ecg_events[:, 0], dtype=int)

    if len(event_samples) < min_events:
        raise RuntimeError(
            f"Only {len(event_samples)} synthetic ECG events were found; "
            f"need at least {min_events}."
        )

    ibi = np.diff(event_samples) / sfreq
    ibi = ibi[np.isfinite(ibi) & (ibi > 0)]

    if len(ibi) == 0:
        raise RuntimeError("No valid synthetic ECG inter-beat intervals.")

    median_bpm = 60.0 / np.median(ibi)
    if not min_bpm <= median_bpm <= max_bpm:
        raise RuntimeError(
            f"Synthetic ECG rate={median_bpm:.1f} bpm is outside "
            f"[{min_bpm}, {max_bpm}] bpm."
        )

    sources = ica.get_sources(raw).get_data(
        reject_by_annotation="NaN"
    )
    n_comp, n_times = sources.shape

    n_pre = int(round(-tmin * sfreq))
    n_post = int(round(tmax * sfreq))
    times = np.arange(-n_pre, n_post + 1) / sfreq

    baseline_mask = (
        (times >= baseline[0]) &
        (times <= baseline[1])
    )
    if baseline_mask.sum() < 3:
        raise ValueError("Baseline interval is too short.")

    event_samples = event_samples[
        (event_samples - n_pre >= 0) &
        (event_samples + n_post < n_times)
    ]

    finite_samples = np.isfinite(sources).all(axis=0)
    event_samples = np.array(
        [
            e for e in event_samples
            if finite_samples[e - n_pre:e + n_post + 1].all()
        ],
        dtype=int,
    )

    if len(event_samples) < min_events:
        raise RuntimeError(
            f"Only {len(event_samples)} usable R-locked epochs remain."
        )

    epochs = np.stack(
        [
            sources[:, e - n_pre:e + n_post + 1]
            for e in event_samples
        ],
        axis=0,
    )
    # Shape: n_events × n_components × n_times_epoch.

    epochs -= epochs[:, :, baseline_mask].mean(axis=2, keepdims=True)

    mean_wave = epochs.mean(axis=0)
    total_var = epochs.var(axis=(0, 2), ddof=1)
    locked_var = mean_wave.var(axis=1, ddof=1)

    r2_lock = locked_var / np.maximum(
        total_var,
        np.finfo(float).eps,
    )
    r2_lock = np.clip(r2_lock, 0.0, 1.0)

    # Stable odd/even R-locked waveform metric.
    odd_mean = epochs[::2].mean(axis=0)
    even_mean = epochs[1::2].mean(axis=0)

    split_r = np.zeros(n_comp)
    for comp in range(n_comp):
        x = odd_mean[comp] - odd_mean[comp].mean()
        y = even_mean[comp] - even_mean[comp].mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        split_r[comp] = (
            np.abs(np.dot(x, y) / denom)
            if denom > np.finfo(float).eps
            else 0.0
        )

    # Use the strongest stable event-locked component as a template.
    # r2 + reproducibility avoids choosing a transient/noisy component.
    template_idx = int(np.argmax(r2_lock * split_r))
    template = mean_wave[template_idx] - mean_wave[template_idx].mean()
    template_norm = np.linalg.norm(template)

    template_r = np.zeros(n_comp)
    for comp in range(n_comp):
        x = mean_wave[comp] - mean_wave[comp].mean()
        denom = np.linalg.norm(x) * template_norm
        template_r[comp] = (
            np.abs(np.dot(x, template) / denom)
            if denom > np.finfo(float).eps
            else 0.0
        )

    # Bounded composite. Add eps to prevent numerical zero underflow.
    cardiac_score = np.cbrt(
        np.maximum(r2_lock, 0.0)
        * np.maximum(split_r, 0.0)
        * np.maximum(template_r, 0.0)
    )

    # Fixed binary decision.
    reject = (
        (cardiac_score >= reject_threshold)
         | (r2_lock >= min_r2_lock)
         | (split_r >= min_split_r)
    )

    results = pd.DataFrame(
        {
            "component": np.arange(n_comp),
            "r2_lock": r2_lock,
            "split_half_abs_r": split_r,
            "template_abs_r": template_r,
            "cardiac_score": cardiac_score,
            "reject": reject,
        }
    ).set_index("component")

    results = results.sort_values(
        "cardiac_score",
        ascending=False,
    )

    qc = {
        "ecg_events": ecg_events,
        "usable_event_samples": event_samples,
        "n_events": len(event_samples),
        "median_bpm": median_bpm,
        "template_component": template_idx,
        "times": times,
        "mean_wave": mean_wave,
    }

    return results, qc