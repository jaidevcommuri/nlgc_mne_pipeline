from __future__ import annotations

import pathlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mne
import numpy as np
import os
import re
from matplotlib.gridspec import GridSpec
from nlgc.utils.leadfield import prepare_eigenmodes


# Must be set before the first MNE 3-D visualization is created.
# For a local desktop pipeline, pyvistaqt is generally the most robust choice.
os.environ.setdefault("MNE_3D_BACKEND", "pyvistaqt")
os.environ.setdefault("MNE_3D_OPTION_MULTI_SAMPLES", "1")


COREG_VIEWS = (
    ("Left", 0, 90),
    ("Left-anterior", 45, 90),
    ("Anterior", 90, 90),
    ("Right-anterior", 135, 90),
    ("Right", 180, 90),
    ("Superior", 90, 0),
)


def _add_message_page(
    pdf: PdfPages,
    title: str,
    message: str,
    *,
    figsize: tuple[float, float] = (16, 9),
) -> None:
    """Add a readable single-page error/missing-data entry to a PDF."""
    fig = plt.figure(figsize=figsize)
    fig.text(0.5, 0.65, title, ha="center", va="center",
             fontsize=22, fontweight="bold")
    fig.text(0.5, 0.45, message, ha="center", va="center",
             fontsize=13, wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _capture_coreg_views(
    raw_info: mne.Info,
    trans: mne.transforms.Transform,
    subject: str,
    subjects_dir: pathlib.Path,
    *,
    distance: float = 0.75,
    image_size: tuple[int, int] = (700, 700),
) -> list[tuple[str, np.ndarray]]:
    """Render six static images from one MNE alignment scene."""
    scene = mne.viz.plot_alignment(
        raw_info,
        trans=trans,
        subject=subject,
        subjects_dir=subjects_dir,
        surfaces="head",
        dig=True,
        eeg=False,
        meg="sensors",
        coord_frame="meg",
        show_axes=True,
    )

    try:
        plotter = scene.plotter

        # Explicitly render before screenshots. This is important outside Jupyter.
        plotter.render()

        images = []
        for label, azimuth, elevation in COREG_VIEWS:
            mne.viz.set_3d_view(
                scene,
                azimuth=azimuth,
                elevation=elevation,
                distance=distance,
                focalpoint=(0.0, 0.0, 0.075),
            )
            plotter.render()
            image = plotter.screenshot(
                return_img=True,
                window_size=image_size,
            )
            images.append((label, image))

        return images
    finally:
        # Always release the VTK/PyVista window/resources.
        scene.plotter.close()


def _coreg_metrics(
    raw_info: mne.Info,
    trans: mne.transforms.Transform,
    subject: str,
    subjects_dir: pathlib.Path,
) -> dict[str, float | int]:
    """Compute concise, interpretable coregistration diagnostics."""
    dev_head_distance_mm = (
        1000 * np.linalg.norm(raw_info["dev_head_t"]["trans"][:3, 3])
    )
    head_mri_distance_mm = (
        1000 * np.linalg.norm(trans["trans"][:3, 3])
    )

    distances_m = mne.dig_mri_distances(
        raw_info,
        trans,
        subject=subject,
        subjects_dir=subjects_dir,
    )
    distances_mm = 1000 * np.asarray(distances_m)

    return {
        "dev_head_mm": dev_head_distance_mm,
        "head_mri_mm": head_mri_distance_mm,
        "n_dig": len(distances_mm),
        "dig_mean_mm": float(np.mean(distances_mm)),
        "dig_median_mm": float(np.median(distances_mm)),
        "dig_max_mm": float(np.max(distances_mm)),
        "dig_p95_mm": float(np.percentile(distances_mm, 95)),
    }


def _add_coreg_page(
    pdf: PdfPages,
    *,
    subject: str,
    raw_info: mne.Info,
    trans: mne.transforms.Transform,
    subjects_dir: pathlib.Path,
    distance: float = 0.75,
    image_size: tuple[int, int] = (700, 700),
) -> None:
    """Generate and append one subject/session coregistration report page."""
    images = _capture_coreg_views(
        raw_info=raw_info,
        trans=trans,
        subject=subject,
        subjects_dir=subjects_dir,
        distance=distance,
        image_size=image_size,
    )
    metrics = _coreg_metrics(
        raw_info=raw_info,
        trans=trans,
        subject=subject,
        subjects_dir=subjects_dir,
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=False)

    for ax, (label, image) in zip(axes.flat, images):
        ax.imshow(image)
        ax.set_title(label, fontsize=11)
        ax.set_axis_off()

    fig.suptitle(f"Coregistration diagnostic: {subject}",
                 fontsize=20, fontweight="bold", y=0.97)

    metric_text = (
        f"Head origin → MEG device origin: "
        f"{metrics['dev_head_mm']:.1f} mm\n"
        f"Head origin → MRI origin: "
        f"{metrics['head_mri_mm']:.1f} mm\n"
        f"Digitized points → scalp surface "
        f"(n={metrics['n_dig']}): "
        f"mean {metrics['dig_mean_mm']:.1f} mm | "
        f"median {metrics['dig_median_mm']:.1f} mm | "
        f"95th percentile {metrics['dig_p95_mm']:.1f} mm | "
        f"maximum {metrics['dig_max_mm']:.1f} mm"
    )

    fig.text(
        0.5,
        0.015,
        metric_text,
        ha="center",
        va="bottom",
        fontsize=11,
        linespacing=1.6,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.13,
                        wspace=0.02, hspace=0.08)

    pdf.savefig(fig)
    plt.close(fig)


def make_coregistration_pdf(
    subjects: Iterable[str],
    config,
    *,
    output_name: str = "coregistration_diagnostics.pdf",
    overwrite: Optional[bool] = None,
    distance: float = 0.75,
    image_size: tuple[int, int] = (700, 700),
) -> pathlib.Path:
    """
    Create a multipage coregistration PDF.

    One PDF page is made for each subject/session. Missing or failed entries
    are retained as labeled pages, which makes the PDF a useful QC manifest.

    Expected paths
    --------------
    Raw:
        {config.data_src.megdir}/{sub}/{sub}_{session}-raw.fif}

    Transform:
        {config.data_src.megdir}/{sub}/{sub}-trans.fif}

    MRI subjects directory:
        config.data_src.mridir
    """
    megdir = pathlib.Path(config.data_src.megdir)
    subjects_dir = pathlib.Path(config.data_src.mridir)
    outdir = pathlib.Path(config.data_src.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_path = outdir / output_name

    if overwrite is None:
        overwrite = config.data_src.overwrite

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Coregistration PDF already exists and overwrite=False: "
            f"{output_path}"
        )

    # Set explicitly rather than relying on whichever backend Jupyter selected.
    mne.viz.set_3d_backend("pyvistaqt")

    with PdfPages(output_path) as pdf:
        for sub in subjects:
            raw_path = megdir / sub / (
                f"{sub}_{config.scan_info.session}-raw.fif"
            )
            trans_path = megdir / sub / f"{sub}-trans.fif"

            if not raw_path.exists():
                _add_message_page(
                    pdf,
                    title=f"MISSING RAW FILE: {sub}",
                    message=str(raw_path),
                )
                continue

            if not trans_path.exists():
                _add_message_page(
                    pdf,
                    title=f"MISSING TRANSFORM FILE: {sub}",
                    message=str(trans_path),
                )
                continue

            subject_mri_dir = subjects_dir / sub
            if not subject_mri_dir.exists():
                _add_message_page(
                    pdf,
                    title=f"MISSING MRI SUBJECT DIRECTORY: {sub}",
                    message=str(subject_mri_dir),
                )
                continue

            try:
                # No sample data are needed for coregistration visualization.
                # preload=False avoids reading all raw MEG samples into memory.
                raw = mne.io.read_raw_fif(
                    raw_path,
                    preload=False,
                    verbose="ERROR",
                )
                trans = mne.read_trans(trans_path, verbose="ERROR")

                _add_coreg_page(
                    pdf,
                    subject=sub,
                    raw_info=raw.info,
                    trans=trans,
                    subjects_dir=subjects_dir,
                    distance=distance,
                    image_size=image_size,
                )

                raw.close()

            except Exception as err:
                _add_message_page(
                    pdf,
                    title=f"COREGISTRATION PLOT FAILED: {sub}",
                    message=f"{type(err).__name__}: {err}",
                )

    return output_path

################################################################################
# Sensor QC
################################################################################


@dataclass(frozen=True)
class SensorQCSettings:
    """Static rendering and PDF layout options."""
    display_sfreq: float = 50.0
    n_rows: int = 3
    n_cols: int = 2
    figsize: tuple[float, float] = (17, 11)

    # Trace appearance
    normal_color: str = "black"
    bad_channel_color: str = "crimson"
    bad_annotation_color: str = "red"
    bad_annotation_alpha: float = 0.16
    normal_linewidth: float = 0.45
    bad_channel_linewidth: float = 0.65

    # Sensor selection and labels
    meg_only: bool = True
    show_every_nth_channel: int = 10

    # Spacing is based on median channel peak-to-peak amplitude.
    # This fallback is in the data's native units, typically tesla for MEG.
    minimum_spacing: float = 1e-15

    # Optional display restriction, e.g. 20 for the first 20 seconds.
    max_duration_s: Optional[float] = None


def _trial_file_stem(
    sub: str,
    session: str,
    trial: int,
    config,
) -> str:
    """
    Return the common filename stem for a trial evoked and annotations.

    This must match the naming in make_evoked().
    """
    l_freq = config.filter_params.analysis_lower_bandlimit
    h_freq = config.filter_params.analysis_upper_bandlimit

    return (
        f"{sub}_{session}"
        f"-[trial={trial}]"
        f"-[{l_freq}-{h_freq}Hz]"
    )


def _trial_evoked_path(
    sub: str,
    session: str,
    trial: int,
    config,
) -> pathlib.Path:
    """Path to one processed trial Evoked FIF file."""
    stem = _trial_file_stem(sub, session, trial, config)

    return (
        pathlib.Path(config.data_src.megdir)
        / sub
        / f"{stem}-evoked-ave.fif"
    )


def _trial_annotations_path(
    sub: str,
    session: str,
    trial: int,
    config,
) -> pathlib.Path:
    """Path to trial-relative annotation sidecar CSV."""
    stem = _trial_file_stem(sub, session, trial, config)

    return (
        pathlib.Path(config.data_src.megdir)
        / sub
        / f"{stem}-annotations.csv"
    )


def _empty_annotations() -> mne.Annotations:
    """Return a valid empty MNE Annotations object."""
    return mne.Annotations(
        onset=[],
        duration=[],
        description=[],
        orig_time=None,
    )


def _load_bad_annotations(
    annotation_path: pathlib.Path,
) -> mne.Annotations:
    """
    Read a per-trial annotation sidecar and retain BAD_* intervals only.

    A missing annotation sidecar does not prevent displaying the evoked:
    it simply yields no temporal red masks.
    """
    if not annotation_path.exists():
        return _empty_annotations()
    
    try:
        annotations = mne.read_annotations(annotation_path)
    except Exception as e:
        return _empty_annotations()

    is_bad = np.array(
        [
            str(description).upper().startswith("BAD")
            for description in annotations.description
        ],
        dtype=bool,
    )

    return annotations[is_bad]


def _sensor_picks(
    evoked: mne.Evoked,
    *,
    meg_only: bool,
) -> np.ndarray:
    """
    Return all desired sensors, including channels in info['bads'].

    `exclude=[]` is essential: otherwise MNE's default behavior can remove
    channels listed in info['bads'], preventing them from being shown in red.
    """
    if meg_only:
        picks = mne.pick_types(
            evoked.info,
            meg=True,
            eeg=False,
            eog=False,
            ecg=False,
            emg=False,
            stim=False,
            misc=False,
            exclude=[],
        )
    else:
        picks = mne.pick_types(
            evoked.info,
            meg=True,
            eeg=True,
            eog=False,
            ecg=False,
            emg=False,
            stim=False,
            misc=False,
            exclude=[],
        )

    if len(picks) == 0:
        raise RuntimeError(
            "No MEG/EEG sensor channels were selected. "
            "Check channel types and the meg_only setting."
        )

    return np.asarray(picks, dtype=int)


def _trace_spacing(
    data: np.ndarray,
    *,
    minimum_spacing: float,
) -> float:
    """Find robust vertical separation for stacked sensor traces."""
    peak_to_peak = np.ptp(data, axis=1)
    usable = peak_to_peak[np.isfinite(peak_to_peak) & (peak_to_peak > 0)]

    if usable.size == 0:
        return minimum_spacing

    return max(2.0 * np.median(usable), minimum_spacing)


def _visible_bad_spans(
    annotations: mne.Annotations,
    *,
    tmin: float,
    tmax: float,
) -> list[tuple[float, float, str]]:
    """Clip trial-relative BAD annotations to the evoked display interval."""
    spans: list[tuple[float, float, str]] = []

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):
        annotation_start = float(onset)
        annotation_stop = annotation_start + float(duration)

        visible_start = max(annotation_start, tmin)
        visible_stop = min(annotation_stop, tmax)

        if visible_stop > visible_start:
            spans.append(
                (
                    visible_start,
                    visible_stop,
                    str(description),
                )
            )

    return spans


def _plot_trial_panel(
    ax: plt.Axes,
    *,
    evoked: mne.Evoked,
    annotations: mne.Annotations,
    sub: str,
    session: str,
    trial: int,
    settings: SensorQCSettings,
) -> None:
    """Draw one stacked-trace trial QC panel."""
    display_evoked = evoked.copy()

    if display_evoked.info["sfreq"] > settings.display_sfreq:
        display_evoked.resample(settings.display_sfreq)

    picks = _sensor_picks(
        display_evoked,
        meg_only=settings.meg_only,
    )

    data = display_evoked.get_data(picks=picks)
    times = display_evoked.times

    if settings.max_duration_s is not None:
        keep = times <= (times[0] + settings.max_duration_s)
        data = data[:, keep]
        times = times[keep]

    if times.size == 0:
        raise RuntimeError("No samples remain after applying max_duration_s.")

    spacing = _trace_spacing(
        data,
        minimum_spacing=settings.minimum_spacing,
    )
    offsets = np.arange(len(picks)) * spacing

    tmin = float(times[0])
    tmax = float(times[-1])

    # Draw the red masks underneath the sensor traces.
    bad_spans = _visible_bad_spans(
        annotations,
        tmin=tmin,
        tmax=tmax,
    )

    for start, stop, _description in bad_spans:
        ax.axvspan(
            start,
            stop,
            color=settings.bad_annotation_color,
            alpha=settings.bad_annotation_alpha,
            linewidth=0,
            zorder=1,
        )

    bad_channel_names = set(display_evoked.info["bads"])
    channel_names = [display_evoked.ch_names[pick] for pick in picks]

    for channel_number, (channel_name, trace, offset) in enumerate(
        zip(channel_names, data, offsets)
    ):
        is_bad_channel = channel_name in bad_channel_names

        ax.plot(
            times,
            trace + offset,
            color=(
                settings.bad_channel_color
                if is_bad_channel
                else settings.normal_color
            ),
            linewidth=(
                settings.bad_channel_linewidth
                if is_bad_channel
                else settings.normal_linewidth
            ),
            zorder=2,
        )

    tick_step = max(1, settings.show_every_nth_channel)
    tick_indices = np.arange(0, len(channel_names), tick_step)

    ax.set_yticks(offsets[tick_indices])
    ax.set_yticklabels(
        [channel_names[index] for index in tick_indices],
        fontsize=6,
    )

    ax.set_xlim(tmin, tmax)
    ax.set_xlabel("Trial-relative time (s)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="0.85", linewidth=0.5, zorder=0)

    # First listed sensor appears at the top.
    ax.invert_yaxis()

    n_bad_channels = sum(
        name in bad_channel_names
        for name in channel_names
    )

    ax.set_title(
        f"{session} | trial {trial} | "
        f"{len(channel_names)} sensors | "
        f"{n_bad_channels} bad channels | "
        f"{len(bad_spans)} BAD spans",
        fontsize=9,
        loc="left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_status_panel(
    ax: plt.Axes,
    *,
    title: str,
    message: str,
    color: str = "0.25",
) -> None:
    """Display a clear non-data panel for requested-but-unavailable data."""
    ax.set_axis_off()
    ax.text(
        0.5,
        0.58,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=color,
    )
    ax.text(
        0.5,
        0.38,
        message,
        ha="center",
        va="center",
        fontsize=7,
        wrap=True,
        color=color,
    )


def _write_subject_page(
    pdf: PdfPages,
    *,
    sub: str,
    records: Sequence[dict],
    settings: SensorQCSettings,
    page_number: int,
) -> None:
    """Write one page containing up to n_rows * n_cols trial panels."""
    fig, axes = plt.subplots(
        settings.n_rows,
        settings.n_cols,
        figsize=settings.figsize,
        squeeze=False,
    )

    for ax, record in zip(axes.flat, records):
        if record["status"] == "missing":
            _plot_status_panel(
                ax,
                title=(
                    f"Missing evoked: {record['session']} | "
                    f"trial {record['trial']}"
                ),
                message=str(record["evoked_path"]),
            )

        elif record["status"] == "failed":
            _plot_status_panel(
                ax,
                title=(
                    f"Failed: {record['session']} | "
                    f"trial {record['trial']}"
                ),
                message=record["error"],
                color="crimson",
            )

        else:
            _plot_trial_panel(
                ax,
                evoked=record["evoked"],
                annotations=record["annotations"],
                sub=sub,
                session=record["session"],
                trial=record["trial"],
                settings=settings,
            )

    # Hide unused slots on the final page for this subject.
    for ax in axes.flat[len(records):]:
        ax.set_axis_off()

    sessions_on_page = sorted(
        {record["session"] for record in records}
    )

    fig.suptitle(
        f"Sensor-data QC: {sub} | page {page_number} | "
        f"sessions: {', '.join(sessions_on_page)}",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.008,
        "Black trace: retained channel    "
        "Red trace: channel in info['bads']    "
        "Translucent red band: BAD_* annotation interval",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.93,
        bottom=0.06,
        hspace=0.48,
        wspace=0.18,
    )

    pdf.savefig(fig)
    plt.close(fig)


def make_sensor_diagnostics_pdf(
    subjects: Iterable[str],
    sessions: Iterable[str],
    trials: Iterable[int],
    config,
    *,
    output_name: str = "sensor_diagnostics.pdf",
    settings: Optional[SensorQCSettings] = None,
    overwrite: Optional[bool] = None,
    include_missing: bool = False,
) -> pathlib.Path:
    """
    Write a multipage processed-sensor diagnostics PDF.

    Each subject occupies one or more consecutive pages. Requested sessions
    and trials are searched independently, so subjects may have arbitrary
    subsets of the session list.

    Each valid trial panel contains:
    - stacked processed MEG traces;
    - channels in evoked.info['bads'] shown in red;
    - translucent red spans for trial-relative BAD_* annotations.

    Annotation files are optional. If the evoked exists but its sidecar
    annotation file is missing, traces are still plotted without red masks.

    Parameters
    ----------
    subjects
        Subject IDs, for example ["R2306", "R2307"].

    sessions
        Session strings used in make_evoked(), for example
        ["CocktailSZ", "SpeechInNoise"].

    trials
        Trial indices, such as range(0, 12) or range(1, 13). This must match
        the numbering used when calling make_evoked().

    config
        Pipeline configuration with `data_src.megdir`, `data_src.outdir`,
        `data_src.overwrite`, and `filter_params` attributes.

    output_name
        PDF filename to write within `config.data_src.outdir`.

    settings
        Optional display/layout configuration.

    overwrite
        Overrides `config.data_src.overwrite` when specified.

    include_missing
        False: silently skip missing evoked files.
        True: reserve labeled blank panels for missing requested trials.

    Returns
    -------
    pathlib.Path
        Location of the generated PDF.
    """
    if settings is None:
        settings = SensorQCSettings()

    if overwrite is None:
        overwrite = config.data_src.overwrite

    subjects = list(subjects)
    sessions = list(sessions)
    trials = list(trials)

    if not subjects:
        raise ValueError("subjects cannot be empty.")
    if not sessions:
        raise ValueError("sessions cannot be empty.")
    if not trials:
        raise ValueError("trials cannot be empty.")

    outdir = pathlib.Path(config.data_src.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / output_name

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists and overwrite=False: {output_path}"
        )

    panels_per_page = settings.n_rows * settings.n_cols

    with PdfPages(output_path) as pdf:
        for sub in subjects:
            subject_records: list[dict] = []

            for session in sessions:
                for trial in trials:
                    evoked_path = _trial_evoked_path(
                        sub=sub,
                        session=session,
                        trial=trial,
                        config=config,
                    )
                    annotations_path = _trial_annotations_path(
                        sub=sub,
                        session=session,
                        trial=trial,
                        config=config,
                    )

                    if not evoked_path.exists():
                        if include_missing:
                            subject_records.append(
                                {
                                    "status": "missing",
                                    "session": session,
                                    "trial": trial,
                                    "evoked_path": evoked_path,
                                    "annotations_path": annotations_path,
                                }
                            )
                        continue

                    try:
                        evoked = mne.read_evokeds(
                            evoked_path,
                            condition=0,
                            verbose="ERROR",
                        )

                        annotations = _load_bad_annotations(
                            annotations_path
                        )

                        subject_records.append(
                            {
                                "status": "ok",
                                "session": session,
                                "trial": trial,
                                "evoked_path": evoked_path,
                                "annotations_path": annotations_path,
                                "evoked": evoked,
                                "annotations": annotations,
                            }
                        )

                    except Exception as error:
                        subject_records.append(
                            {
                                "status": "failed",
                                "session": session,
                                "trial": trial,
                                "evoked_path": evoked_path,
                                "annotations_path": annotations_path,
                                "error": (
                                    f"{type(error).__name__}: {error}"
                                ),
                            }
                        )

            if not subject_records:
                fig = plt.figure(figsize=(11, 8.5))

                fig.text(
                    0.5,
                    0.60,
                    f"No requested processed trials found for {sub}",
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                )
                fig.text(
                    0.5,
                    0.43,
                    f"Sessions requested: {', '.join(sessions)}\n"
                    f"Trials requested: {trials}",
                    ha="center",
                    va="center",
                    fontsize=11,
                )

                pdf.savefig(fig)
                plt.close(fig)
                continue

            for start in range(0, len(subject_records), panels_per_page):
                page_records = subject_records[
                    start:start + panels_per_page
                ]

                page_number = (start // panels_per_page) + 1

                _write_subject_page(
                    pdf,
                    sub=sub,
                    records=page_records,
                    settings=settings,
                    page_number=page_number,
                )

    return output_path


################################################################################
# ICA QC
################################################################################


@dataclass(frozen=True)
class ICAQCSettings:
    display_sfreq: float = 50.0

    # Session-window sampling.
    window_duration_s: float = 20.0
    n_windows_per_session: int = 6
    window_selection: str = "even"
    candidate_windows_multiplier: int = 8

    # PDF layout.
    n_rows: int = 3
    n_cols: int = 2
    figsize: tuple[float, float] = (17, 11)

    # Channel selection.
    meg_only: bool = True
    show_every_nth_channel: int = 10
    minimum_spacing: float = 1e-15

    # Colors and line appearance.
    raw_color: str = "black"
    bad_channel_color: str = "darkorange"
    artifact_color: str = "crimson"

    raw_linewidth: float = 0.35
    artifact_linewidth: float = 0.85
    raw_alpha: float = 0.82
    artifact_alpha: float = 0.95

    # Red highlighting threshold.
    artifact_relative_threshold: float = 0.10
    artifact_minimum_amplitude: float = 0.0


def _ica_select_sensor_picks(
    info: mne.Info,
    *,
    meg_only: bool,
) -> np.ndarray:
    """
    Select sensor channels for ICA QC, retaining channels in info['bads'].

    `exclude=[]` is intentional: bad channels should remain visible in the
    diagnostic report and be colored separately.
    """
    if meg_only:
        picks = mne.pick_types(
            info,
            meg=True,
            eeg=False,
            eog=False,
            ecg=False,
            emg=False,
            stim=False,
            misc=False,
            exclude=[],
        )
    else:
        picks = mne.pick_types(
            info,
            meg=True,
            eeg=True,
            eog=False,
            ecg=False,
            emg=False,
            stim=False,
            misc=False,
            exclude=[],
        )

    if len(picks) == 0:
        raise RuntimeError(
            "ICA QC could not find MEG/EEG channels to plot."
        )

    return np.asarray(picks, dtype=int)


def _ica_trace_spacing(
    data: np.ndarray,
    *,
    minimum_spacing: float,
) -> float:
    """Compute robust vertical trace spacing from raw sensor amplitudes."""
    peak_to_peak = np.ptp(data, axis=1)

    usable = peak_to_peak[
        np.isfinite(peak_to_peak) & (peak_to_peak > 0)
    ]

    if usable.size == 0:
        return minimum_spacing

    return max(
        2.0 * np.median(usable),
        minimum_spacing,
    )


def _ica_crop_loaded_window(
    raw: mne.io.BaseRaw,
    *,
    start_s: float,
    duration_s: float,
) -> mne.io.BaseRaw:
    """
    Crop and explicitly load a session-time window.

    `ICA.apply()` requires data to be available. Although `crop()` changes
    the time bounds, it does not necessarily load the corresponding samples
    when the source Raw was opened with preload=False.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be greater than zero.")

    raw_start_s = float(raw.times[0])
    raw_stop_s = float(raw.times[-1])

    if start_s < raw_start_s:
        raise ValueError(
            f"Requested window start ({start_s:.3f} s) precedes "
            f"the raw start ({raw_start_s:.3f} s)."
        )

    stop_s = min(start_s + duration_s, raw_stop_s)

    if stop_s <= start_s:
        raise ValueError(
            f"Invalid ICA QC window: {start_s:.3f}–{stop_s:.3f} s."
        )

    return raw.copy().crop(
        tmin=start_s,
        tmax=stop_s,
    ).load_data()


def _ica_make_artifact_window(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    *,
    start_s: float,
    settings: ICAQCSettings,
) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw, mne.io.BaseRaw]:
    """
    Return original, ICA-cleaned, and removed-artifact windows.

    The artifact signal is calculated in sensor space:

        removed_artifact = original_pre_ICA - ICA_cleaned

    `exclude=None` tells MNE to use the saved `ica.exclude` list.
    """
    raw_window = _ica_crop_loaded_window(
        raw,
        start_s=start_s,
        duration_s=settings.window_duration_s,
    )

    cleaned_window = raw_window.copy()

    ica.apply(
        cleaned_window,
        exclude=None,
        verbose="ERROR",
    )

    artifact_window = raw_window.copy()

    artifact_window._data = (
        raw_window.get_data()
        - cleaned_window.get_data()
    )

    # Preserve the original session-window timing metadata as much as
    # possible; the actual x-axis is constructed separately below.
    artifact_window.preload = True

    # Resample only after calculating the difference so all three versions
    # begin from exactly the same time points and channel values.
    if raw_window.info["sfreq"] > settings.display_sfreq:
        raw_window.resample(
            settings.display_sfreq,
            npad="auto",
        )
        cleaned_window.resample(
            settings.display_sfreq,
            npad="auto",
        )
        artifact_window.resample(
            settings.display_sfreq,
            npad="auto",
        )

    return raw_window, cleaned_window, artifact_window


def _ica_artifact_active_mask(
    artifact_trace: np.ndarray,
    *,
    channel_raw_ptp: float,
    settings: ICAQCSettings,
) -> np.ndarray:
    """
    Identify samples with a sufficiently large removed ICA contribution.

    Threshold is relative to the raw channel's peak-to-peak range within the
    current display window. This preserves sensitivity to localized artifacts
    while preventing nearly-zero differences from producing universal red
    overlays.
    """
    channel_scale = max(
        float(channel_raw_ptp),
        settings.minimum_spacing,
    )

    threshold = max(
        settings.artifact_minimum_amplitude,
        settings.artifact_relative_threshold * channel_scale,
    )

    return np.abs(artifact_trace) >= threshold


def _ica_plot_window_panel(
    ax: plt.Axes,
    *,
    raw_window: mne.io.BaseRaw,
    artifact_window: mne.io.BaseRaw,
    sub: str,
    session: str,
    window_start_s: float,
    excluded_components: Sequence[int],
    settings: ICAQCSettings,
) -> None:
    """
    Plot one ICA QC window.

    Normal raw traces are black. Raw traces from channels listed in
    `info['bads']` are orange. Red segments mark samples at which the removed
    ICA contribution exceeds the configured threshold.
    """
    picks = _ica_select_sensor_picks(
        raw_window.info,
        meg_only=settings.meg_only,
    )

    raw_data = raw_window.get_data(picks=picks)
    artifact_data = artifact_window.get_data(picks=picks)

    if raw_data.shape != artifact_data.shape:
        raise RuntimeError(
            "ICA QC raw and artifact windows have incompatible shapes: "
            f"{raw_data.shape} versus {artifact_data.shape}."
        )

    if raw_data.shape[1] == 0:
        raise RuntimeError("ICA QC window contains no samples.")

    local_times = raw_window.times
    session_times = local_times + window_start_s

    spacing = _ica_trace_spacing(
        raw_data,
        minimum_spacing=settings.minimum_spacing,
    )
    offsets = np.arange(len(picks), dtype=float) * spacing

    channel_names = [raw_window.ch_names[pick] for pick in picks]
    bad_channel_names = set(raw_window.info["bads"])

    raw_peak_to_peak = np.ptp(raw_data, axis=1)

    for channel_index, (
        channel_name,
        raw_trace,
        artifact_trace,
        offset,
    ) in enumerate(
        zip(
            channel_names,
            raw_data,
            artifact_data,
            offsets,
        )
    ):
        is_bad_channel = channel_name in bad_channel_names

        base_color = (
            settings.bad_channel_color
            if is_bad_channel
            else settings.raw_color
        )

        # Full original trace.
        ax.plot(
            session_times,
            raw_trace + offset,
            color=base_color,
            linewidth=settings.raw_linewidth,
            alpha=settings.raw_alpha,
            zorder=2,
        )

        # Plot the original trace in red only during samples where the
        # excluded ICA contribution is sufficiently strong.
        artifact_active = _ica_artifact_active_mask(
            artifact_trace,
            channel_raw_ptp=raw_peak_to_peak[channel_index],
            settings=settings,
        )

        if np.any(artifact_active):
            red_raw_trace = np.where(
                artifact_active,
                raw_trace + offset,
                np.nan,
            )

            ax.plot(
                session_times,
                red_raw_trace,
                color=settings.artifact_color,
                linewidth=settings.artifact_linewidth,
                alpha=settings.artifact_alpha,
                zorder=3,
            )

    tick_step = max(1, settings.show_every_nth_channel)
    tick_indices = np.arange(0, len(channel_names), tick_step)

    ax.set_yticks(offsets[tick_indices])
    ax.set_yticklabels(
        [channel_names[index] for index in tick_indices],
        fontsize=6,
    )

    ax.invert_yaxis()

    session_stop_s = float(session_times[-1])

    ax.set_xlim(
        float(session_times[0]),
        session_stop_s,
    )
    ax.set_xlabel("Session time (s)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(
        axis="x",
        color="0.85",
        linewidth=0.5,
        zorder=0,
    )

    excluded_label = (
        ", ".join(str(index) for index in excluded_components)
        if excluded_components
        else "none"
    )

    n_channels_with_artifact = int(
        np.sum(
            [
                np.any(
                    _ica_artifact_active_mask(
                        artifact_data[channel_index],
                        channel_raw_ptp=raw_peak_to_peak[channel_index],
                        settings=settings,
                    )
                )
                for channel_index in range(len(channel_names))
            ]
        )
    )

    ax.set_title(
        f"{session} | {window_start_s:.1f}-{session_stop_s:.1f} s | "
        f"excluded ICs: {excluded_label} | "
        f"artifact visible: {n_channels_with_artifact}/{len(channel_names)}",
        fontsize=8.3,
        loc="left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ica_file_stem(
    sub: str,
    session: str,
    config,
) -> str:
    """
    Shared filename stem for pre-ICA raw and ICA decomposition files.

    Matches:
    {sub}_tsss-{l_filt}-{h_filt}-{session}-raw.fif
    {sub}_tsss-{l_filt}-{h_filt}-{session}-apply-comp-ica.fif
    """
    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit

    return f"{sub}_tsss-{l_filt}-{h_filt}-{session}"


def _ica_raw_path(
    sub: str,
    session: str,
    config,
) -> pathlib.Path:
    """Return the session-level tSSS raw file preceding ICA application."""
    stem = _ica_file_stem(
        sub=sub,
        session=session,
        config=config,
    )

    return (
        pathlib.Path(config.data_src.megdir)
        / sub
        / f"{stem}-raw.fif"
    )


def _ica_solution_path(
    sub: str,
    session: str,
    config,
) -> pathlib.Path:
    """Return the saved ICA decomposition and exclusion-list file."""
    stem = _ica_file_stem(
        sub=sub,
        session=session,
        config=config,
    )

    return (
        pathlib.Path(config.data_src.megdir)
        / sub
        / f"{stem}-apply-comp-ica.fif"
    )


def _ica_even_window_starts(
    raw: mne.io.BaseRaw,
    *,
    duration_s: float,
    n_windows: int,
) -> np.ndarray:
    """Select evenly distributed valid window starts across a session."""
    if duration_s <= 0:
        raise ValueError("ICA QC window_duration_s must be positive.")

    if n_windows <= 0:
        raise ValueError("ICA QC n_windows_per_session must be positive.")

    raw_start_s = float(raw.times[0])
    raw_stop_s = float(raw.times[-1])
    final_start_s = raw_stop_s - duration_s

    if final_start_s <= raw_start_s:
        return np.array([raw_start_s], dtype=float)

    return np.linspace(
        raw_start_s,
        final_start_s,
        num=n_windows,
        dtype=float,
    )


def _ica_artifact_window_score(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    *,
    start_s: float,
    settings: ICAQCSettings,
) -> float:
    """
    Score one window by removed ICA artifact amplitude in sensor space.

    Higher values indicate a larger median channel peak-to-peak contribution
    from the components in `ica.exclude`.
    """
    raw_window, _cleaned_window, artifact_window = (
        _ica_make_artifact_window(
            raw=raw,
            ica=ica,
            start_s=start_s,
            settings=settings,
        )
    )

    picks = _ica_select_sensor_picks(
        raw_window.info,
        meg_only=settings.meg_only,
    )

    artifact_data = artifact_window.get_data(picks=picks)

    if artifact_data.shape[1] == 0:
        return 0.0

    return float(np.median(np.ptp(artifact_data, axis=1)))


def _ica_choose_window_starts(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    *,
    settings: ICAQCSettings,
) -> np.ndarray:
    """
    Select report windows using even or high-artifact sampling.

    "even"
        Spread windows across the entire session.

    "largest_artifact"
        Score a denser grid of candidate windows by the sensor-space magnitude
        removed by ICA, select the strongest windows, then order them in time.
    """
    if settings.window_selection == "even":
        return _ica_even_window_starts(
            raw,
            duration_s=settings.window_duration_s,
            n_windows=settings.n_windows_per_session,
        )

    if settings.window_selection != "largest_artifact":
        raise ValueError(
            "ICAQCSettings.window_selection must be 'even' or "
            "'largest_artifact'."
        )

    raw_start_s = float(raw.times[0])
    raw_stop_s = float(raw.times[-1])
    final_start_s = raw_stop_s - settings.window_duration_s

    if final_start_s <= raw_start_s:
        return np.array([raw_start_s], dtype=float)

    n_candidates = max(
        settings.n_windows_per_session,
        settings.n_windows_per_session
        * settings.candidate_windows_multiplier,
    )

    candidate_starts = np.linspace(
        raw_start_s,
        final_start_s,
        num=n_candidates,
        dtype=float,
    )

    scores = np.array(
        [
            _ica_artifact_window_score(
                raw=raw,
                ica=ica,
                start_s=float(start_s),
                settings=settings,
            )
            for start_s in candidate_starts
        ],
        dtype=float,
    )

    n_select = min(
        settings.n_windows_per_session,
        len(candidate_starts),
    )

    best_indices = np.argsort(scores)[-n_select:]

    # Chronological output is easier to inspect than score-ranked output.
    return np.sort(candidate_starts[best_indices])


def _ica_plot_status_panel(
    ax: plt.Axes,
    *,
    title: str,
    message: str,
    color: str = "0.25",
) -> None:
    """Display a missing-input or failure message in a report panel."""
    ax.set_axis_off()

    ax.text(
        0.5,
        0.60,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=color,
    )

    ax.text(
        0.5,
        0.36,
        message,
        ha="center",
        va="center",
        fontsize=7,
        wrap=True,
        color=color,
    )


def _ica_write_qc_page(
    pdf: PdfPages,
    *,
    sub: str,
    records: Sequence[dict],
    page_number: int,
    settings: ICAQCSettings,
) -> None:
    """Render a single multi-panel ICA QC page."""
    fig, axes = plt.subplots(
        settings.n_rows,
        settings.n_cols,
        figsize=settings.figsize,
        squeeze=False,
    )

    for ax, record in zip(axes.flat, records):
        if record["status"] == "failed":
            _ica_plot_status_panel(
                ax,
                title=record["title"],
                message=record["error"],
                color="crimson",
            )
            continue

        _ica_plot_window_panel(
            ax,
            raw_window=record["raw_window"],
            artifact_window=record["artifact_window"],
            sub=sub,
            session=record["session"],
            window_start_s=record["window_start_s"],
            excluded_components=record["excluded_components"],
            settings=settings,
        )

    for ax in axes.flat[len(records):]:
        ax.set_axis_off()

    sessions_on_page = ", ".join(
        sorted(
            {
                record["session"]
                for record in records
            }
        )
    )

    fig.suptitle(
        f"ICA artifact-removal QC: {sub} | page {page_number} | "
        f"sessions: {sessions_on_page}",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.008,
        "Black: original pre-ICA sensor data    "
        "Orange: channel listed in info['bads']    "
        "Red: original trace during a substantial removed ICA contribution",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.93,
        bottom=0.06,
        hspace=0.48,
        wspace=0.18,
    )

    pdf.savefig(fig)
    plt.close(fig)


def _ica_write_no_data_page(
    pdf: PdfPages,
    *,
    sub: str,
    sessions: Sequence[str],
) -> None:
    """Write one explicit page when none of a subject's ICA inputs exist."""
    fig = plt.figure(figsize=(11, 8.5))

    fig.text(
        0.5,
        0.60,
        f"No ICA inputs found for {sub}",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.43,
        f"Requested sessions: {', '.join(sessions)}",
        ha="center",
        va="center",
        fontsize=11,
    )

    pdf.savefig(fig)
    plt.close(fig)


def make_ica_apply_diagnostic_pdf(
    subjects: Iterable[str],
    sessions: Iterable[str],
    config,
    *,
    output_name: str = "ica_artifact_qc.pdf",
    settings: Optional[ICAQCSettings] = None,
    overwrite: Optional[bool] = None,
    include_missing: bool = False,
) -> pathlib.Path:
    """
    Create a multipage ICA artifact-removal QC PDF.

    For every existing subject/session pair, the function loads:

    - The tSSS raw file immediately preceding ICA application:
      {sub}_tsss-{wide_l}-{wide_h}-{session}-raw.fif

    - The saved ICA decomposition and its exclusion list:
      {sub}_tsss-{wide_l}-{wide_h}-{session}-apply-comp-ica.fif

    It samples several 20-second windows from the complete session, applies
    the ICA exclusion list to each loaded window, and calculates:

        removed ICA contribution = raw before ICA - ICA-cleaned raw

    In each panel:
    - black is the original pre-ICA sensor trace;
    - orange is a sensor in `raw.info["bads"]`;
    - red highlights portions of the original trace where the excluded ICA
      contribution exceeds the configured threshold.

    Parameters
    ----------
    subjects
        Iterable of subject IDs.

    sessions
        Iterable of session strings. Each subject may have only a subset.

    config
        Pipeline configuration with `data_src.megdir`, `data_src.outdir`,
        `data_src.overwrite`, and `filter_params` wideband limits.

    output_name
        Filename for the report within `config.data_src.outdir`.

    settings
        ICA report rendering and layout settings.

    overwrite
        If supplied, overrides `config.data_src.overwrite`.

    include_missing
        If False, silently skip nonexistent raw/ICA input pairs.
        If True, include a status panel for missing pairs.

    Returns
    -------
    pathlib.Path
        The generated PDF path.
    """
    if settings is None:
        settings = ICAQCSettings()

    if overwrite is None:
        overwrite = config.data_src.overwrite

    subjects = list(subjects)
    sessions = list(sessions)

    if not subjects:
        raise ValueError("subjects cannot be empty.")

    if not sessions:
        raise ValueError("sessions cannot be empty.")

    outdir = pathlib.Path(config.data_src.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_path = outdir / output_name

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"ICA QC output exists and overwrite=False: {output_path}"
        )

    panels_per_page = settings.n_rows * settings.n_cols

    with PdfPages(output_path) as pdf:
        for sub in subjects:
            subject_records: list[dict] = []

            for session in sessions:
                raw_path = _ica_raw_path(
                    sub=sub,
                    session=session,
                    config=config,
                )
                solution_path = _ica_solution_path(
                    sub=sub,
                    session=session,
                    config=config,
                )

                raw_exists = raw_path.exists()
                ica_exists = solution_path.exists()

                if not raw_exists or not ica_exists:
                    if include_missing:
                        missing_messages = []

                        if not raw_exists:
                            missing_messages.append(
                                f"Missing pre-ICA raw:\n{raw_path}"
                            )

                        if not ica_exists:
                            missing_messages.append(
                                f"Missing ICA solution:\n{solution_path}"
                            )

                        subject_records.append(
                            {
                                "status": "failed",
                                "session": session,
                                "title": (
                                    f"Missing ICA input: {sub} | {session}"
                                ),
                                "error": "\n\n".join(missing_messages),
                            }
                        )

                    continue

                raw = None

                try:
                    # Keep the complete recording lazy. Each selected window
                    # is cropped and explicitly loaded by
                    # _ica_crop_loaded_window().
                    raw = mne.io.read_raw_fif(
                        raw_path,
                        preload=False,
                        verbose="ERROR",
                    )

                    # This is an ICA object saved with `ica.save()`, not a Raw.
                    ica = mne.preprocessing.read_ica(
                        solution_path,
                        verbose="ERROR",
                    )

                    excluded_components = list(ica.exclude)

                    window_starts = _ica_choose_window_starts(
                        raw=raw,
                        ica=ica,
                        settings=settings,
                    )

                    for window_start_s in window_starts:
                        raw_window, _cleaned_window, artifact_window = (
                            _ica_make_artifact_window(
                                raw=raw,
                                ica=ica,
                                start_s=float(window_start_s),
                                settings=settings,
                            )
                        )

                        subject_records.append(
                            {
                                "status": "ok",
                                "session": session,
                                "window_start_s": float(window_start_s),
                                "excluded_components": excluded_components,
                                "raw_window": raw_window,
                                "artifact_window": artifact_window,
                            }
                        )

                except Exception as error:
                    subject_records.append(
                        {
                            "status": "failed",
                            "session": session,
                            "title": f"ICA QC failed: {sub} | {session}",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

                finally:
                    if raw is not None:
                        raw.close()

            if not subject_records:
                _ica_write_no_data_page(
                    pdf,
                    sub=sub,
                    sessions=sessions,
                )
                continue

            for page_start in range(
                0,
                len(subject_records),
                panels_per_page,
            ):
                page_records = subject_records[
                    page_start:page_start + panels_per_page
                ]

                _ica_write_qc_page(
                    pdf,
                    sub=sub,
                    records=page_records,
                    page_number=(page_start // panels_per_page) + 1,
                    settings=settings,
                )

    return output_path

################################################################################
# ICA components QC
################################################################################

@dataclass(frozen=True)
class ICASourcesDiagnosticSettings:
    """Layout settings for ICA source-browser and spectral QC."""
    window_duration_s: float = 20.0
    n_windows_per_session: int = 4

    # "even" gives representative session coverage.
    # "manual" uses `window_starts_s` exactly.
    window_selection: str = "even"
    window_starts_s: Optional[tuple[float, ...]] = None

    # ICA browser figure settings.
    source_figsize: tuple[float, float] = (16, 10)

    # Session-level component spectra.
    psd_fmin: float = 1.0
    psd_fmax: float = 100.0
    psd_n_fft: int = 2048
    psd_figsize: tuple[float, float] = (16, 10)
    psd_n_cols: int = 4
    psd_max_components_per_page: int = 24
    psd_color_good: str = "black"
    psd_color_rejected: str = "crimson"
    psd_linewidth: float = 0.7

def _ica_sources_diagnostic_stem(
    sub: str,
    session: str,
    config,
) -> str:
    """Build the shared raw/ICA filename stem."""
    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit

    return f"{sub}_tsss-{l_filt}-{h_filt}-{session}"


def _ica_sources_diagnostic_raw_path(
    sub: str,
    session: str,
    config,
) -> pathlib.Path:
    """Path to the tSSS raw file directly preceding ICA application."""
    stem = _ica_sources_diagnostic_stem(sub, session, config)

    return (
        pathlib.Path(config.data_src.megdir)
        / sub
        / f"{stem}-raw.fif"
    )


def _ica_sources_diagnostic_ica_path(
    sub: str,
    session: str,
    config,
) -> pathlib.Path:
    """Path to the ICA object saved by `ica.save()`."""
    stem = _ica_sources_diagnostic_stem(sub, session, config)

    return (
        pathlib.Path(config.data_src.megdir)
        / sub
        / f"{stem}-apply-comp-ica.fif"
    )


def _ica_sources_diagnostic_window_starts(
    raw: mne.io.BaseRaw,
    *,
    settings: ICASourcesDiagnosticSettings,
) -> np.ndarray:
    """Choose valid 20-second starts from a session-level raw recording."""
    raw_start_s = float(raw.times[0])
    raw_stop_s = float(raw.times[-1])

    if settings.window_duration_s <= 0:
        raise ValueError("window_duration_s must be positive.")

    if settings.window_selection == "manual":
        if not settings.window_starts_s:
            raise ValueError(
                "window_starts_s must be provided when "
                "window_selection='manual'."
            )

        starts = np.asarray(
            settings.window_starts_s,
            dtype=float,
        )

        return starts[
            (starts >= raw_start_s)
            & (starts < raw_stop_s)
        ]

    if settings.window_selection != "even":
        raise ValueError(
            "window_selection must be 'even' or 'manual'."
        )

    if settings.n_windows_per_session <= 0:
        raise ValueError("n_windows_per_session must be positive.")

    final_start_s = raw_stop_s - settings.window_duration_s

    if final_start_s <= raw_start_s:
        return np.array([raw_start_s], dtype=float)

    return np.linspace(
        raw_start_s,
        final_start_s,
        num=settings.n_windows_per_session,
        dtype=float,
    )

def _ica_sources_diagnostic_save_browser_page(
    pdf: PdfPages,
    *,
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    sub: str,
    session: str,
    start_s: float,
    settings: ICASourcesDiagnosticSettings,
    page_number: int,
    n_windows: int,
) -> None:
    """
    Save one MNE ICA source-browser view to the report PDF.

    MNE handles ICA source traces, exclusions, annotations, and scrolling.
    Do not pass `picks` here: with some MNE versions, explicitly passing
    component indices can trigger ICA plot_sources indexing errors.
    """
    raw_start_s = float(raw.times[0])
    raw_stop_s = float(raw.times[-1])

    stop_s = min(
        float(start_s) + settings.window_duration_s,
        raw_stop_s,
    )
    duration_s = stop_s - float(start_s)

    fig = ica.plot_sources(
        raw,
        start=float(start_s),
        stop=stop_s,
        show=False,
    )

    # MNE may set/re-set its axes title internally. Figure-level text is
    # independent of that title and will be included in the PDF.
    fig.text(
        0.5,
        0.985,
        f"ICA source diagnostic | Subject: {sub} | Session: {session}",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.012,
        f"Window {page_number}/{n_windows} | "
        f"Session interval: {start_s:.2f}–{stop_s:.2f} s | "
        f"Window duration: {duration_s:.2f} s | "
        f"Full recording: {raw_start_s:.2f}–{raw_stop_s:.2f} s\n"
        f"Rejected ICA components: {list(ica.exclude) if ica.exclude else 'none'} | "
        f"Red = rejected; black = retained | "
        f"Native MNE annotations are shown in the source browser.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    # Important: use a large enough bottom margin that the metadata footer
    # does not cover MNE's source-browser scroll/navigation controls.
    fig.subplots_adjust(
        top=0.94,
        bottom=0.11,
    )

    pdf.savefig(fig)
    plt.close(fig)

def _ica_sources_diagnostic_component_index(
    channel_name: str,
) -> int:
    """
    Extract an ICA component index from an MNE ICA source channel name.

    Supported examples:
    - "ICA000"  -> 0
    - "ICA017"  -> 17
    - "ICA 000" -> 0
    - "ICA 017" -> 17
    - "IC_17"   -> 17
    - "component-17" -> 17
    """
    match = re.search(
        r"(\d+)\s*$",
        str(channel_name),
    )

    if match is None:
        raise ValueError(
            "Could not determine ICA component index from source "
            f"channel name: {channel_name!r}"
        )

    return int(match.group(1))
    
    
def _ica_sources_diagnostic_write_psd_pages(
    pdf: PdfPages,
    *,
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    sub: str,
    session: str,
    settings: ICASourcesDiagnosticSettings,
) -> None:
    """
    Write session-level Welch PSD grids for every ICA component.

    Rejected components are red and retained components are black.

    Important:
    - ICA source channels are picked explicitly.
    - `exclude=[]` is passed to both compute_psd() and get_data() so source
      channels marked bad cannot silently disappear and desynchronize PSD-row
      order from the source-channel list.
    """
    source_raw = ica.get_sources(raw)

    source_picks = np.arange(
        len(source_raw.ch_names),
        dtype=int,
    )

    if source_picks.size == 0:
        raise RuntimeError(
            f"No ICA source channels found for {sub} | {session}."
        )

    source_channel_names = [
        source_raw.ch_names[pick]
        for pick in source_picks
    ]

    spectrum = source_raw.compute_psd(
        method="welch",
        fmin=settings.psd_fmin,
        fmax=settings.psd_fmax,
        picks=source_picks,
        exclude=[],
        n_fft=settings.psd_n_fft,
        verbose="ERROR",
    )

    psds, freqs = spectrum.get_data(
        picks=np.arange(len(source_channel_names)),
        exclude=[],
        return_freqs=True,
    )

    if psds.ndim != 2:
        raise RuntimeError(
            "Expected PSD data shaped "
            "(n_components, n_frequencies), got "
            f"{psds.shape}."
        )

    if psds.shape[0] != len(source_channel_names):
        raise RuntimeError(
            "ICA PSD channel mismatch after explicit selections: "
            f"{psds.shape[0]} PSD rows versus "
            f"{len(source_channel_names)} selected source channels. "
            f"PSD channels: {getattr(spectrum, 'ch_names', 'unknown')}"
        )

    psds_db = 10.0 * np.log10(
        np.maximum(
            psds,
            np.finfo(float).tiny,
        )
    )

    rejected_components = {
        int(component)
        for component in ica.exclude
    }

    # PSD row index is intentionally separate from ICA component index.
    component_rows = [
        (
            psd_row,
            _ica_sources_diagnostic_component_index(
                source_channel_names[psd_row]
            ),
            source_channel_names[psd_row],
        )
        for psd_row in range(len(source_channel_names))
    ]

    components_per_page = max(
        1,
        settings.psd_max_components_per_page,
    )
    n_cols = max(1, settings.psd_n_cols)
    n_rows = int(
        np.ceil(components_per_page / n_cols)
    )

    n_pages = int(
        np.ceil(len(component_rows) / components_per_page)
    )

    for page_number, start_index in enumerate(
        range(0, len(component_rows), components_per_page),
        start=1,
    ):
        page_rows = component_rows[
            start_index:start_index + components_per_page
        ]

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=settings.psd_figsize,
            squeeze=False,
            sharex=True,
        )

        for ax, (
            psd_row,
            component_index,
            _source_channel_name,
        ) in zip(
            axes.flat,
            page_rows,
        ):
            is_rejected = component_index in rejected_components

            ax.plot(
                freqs,
                psds_db[psd_row],
                color=(
                    settings.psd_color_rejected
                    if is_rejected
                    else settings.psd_color_good
                ),
                linewidth=settings.psd_linewidth,
            )

            status = (
                "rejected"
                if is_rejected
                else "retained"
            )

            ax.set_title(
                f"IC {component_index}: {status}",
                fontsize=8,
                color=(
                    settings.psd_color_rejected
                    if is_rejected
                    else settings.psd_color_good
                ),
            )

            ax.set_xlim(
                settings.psd_fmin,
                settings.psd_fmax,
            )
            ax.grid(
                color="0.88",
                linewidth=0.5,
            )
            ax.tick_params(
                axis="both",
                labelsize=6,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for ax in axes.flat[len(page_rows):]:
            ax.set_axis_off()

        for ax in axes[-1, :]:
            ax.set_xlabel(
                "Frequency (Hz)",
                fontsize=8,
            )

        for ax in axes[:, 0]:
            ax.set_ylabel(
                "Power (dB)",
                fontsize=8,
            )

        component_ids = [
            component_index
            for _, component_index, _ in page_rows
        ]

        fig.suptitle(
            f"ICA component spectra | Subject: {sub} | "
            f"Session: {session} | PSD page {page_number}/{n_pages}",
            fontsize=15,
            fontweight="bold",
            y=0.985,
        )

        fig.text(
            0.5,
            0.008,
            f"Welch PSD: {settings.psd_fmin:g}–"
            f"{settings.psd_fmax:g} Hz | "
            f"n_fft={settings.psd_n_fft} | "
            f"Components on this page: {component_ids} | "
            f"Rejected/red: {sorted(rejected_components) or 'none'} | "
            f"Retained/black: all others",
            ha="center",
            va="bottom",
            fontsize=8,
        )

        fig.subplots_adjust(
            left=0.07,
            right=0.99,
            top=0.93,
            bottom=0.06,
            hspace=0.62,
            wspace=0.28,
        )

        pdf.savefig(fig)
        plt.close(fig)

def make_ica_sources_diagnostic_pdf(
    subjects: Iterable[str],
    sessions: Iterable[str],
    config,
    *,
    output_name: str = "ica_sources_diagnostic.pdf",
    settings: Optional[ICASourcesDiagnosticSettings] = None,
    overwrite: Optional[bool] = None,
) -> pathlib.Path:
    """
    Create a multipage ICA diagnostic PDF.

    For each available subject/session pair, writes:

    1. Several MNE ICA-source browser pages covering windows distributed
       through the complete raw session. MNE displays the raw annotations
       and component rejection status.

    2. One or more session-level Welch PSD grid pages showing all ICA
       components. Rejected components are red; retained components black.

    The report adds subject, session, exact session-time interval, actual
    duration, recording range, rejected-component list, component index
    range, and PSD settings to the exported pages.
    """
    if settings is None:
        settings = ICASourcesDiagnosticSettings()

    if overwrite is None:
        overwrite = config.data_src.overwrite

    subjects = list(subjects)
    sessions = list(sessions)

    if not subjects:
        raise ValueError("subjects cannot be empty.")

    if not sessions:
        raise ValueError("sessions cannot be empty.")

    if settings.window_duration_s <= 0:
        raise ValueError("window_duration_s must be positive.")

    if settings.psd_fmin < 0:
        raise ValueError("psd_fmin must be non-negative.")

    if settings.psd_fmax <= settings.psd_fmin:
        raise ValueError("psd_fmax must be greater than psd_fmin.")

    outdir = pathlib.Path(config.data_src.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_path = outdir / output_name

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists and overwrite=False: {output_path}"
        )

    # This makes `ica.plot_sources()` return a standard Matplotlib Figure,
    # which can be augmented and passed directly to PdfPages.
    mne.viz.set_browser_backend("matplotlib")

    with PdfPages(output_path) as pdf:
        for sub in subjects:
            found_input_pair = False
            wrote_any_page = False

            for session in sessions:
                raw_path = _ica_sources_diagnostic_raw_path(
                    sub=sub,
                    session=session,
                    config=config,
                )
                ica_path = _ica_sources_diagnostic_ica_path(
                    sub=sub,
                    session=session,
                    config=config,
                )

                if not raw_path.exists() or not ica_path.exists():
                    continue

                # The pair exists, regardless of whether a later plotting action
                # fails. This prevents the incorrect "No ... pairs found" page.
                found_input_pair = True

                raw = None

                try:
                    raw = mne.io.read_raw_fif(
                        raw_path,
                        preload=True,
                        verbose="ERROR",
                    )

                    ica = mne.preprocessing.read_ica(
                        ica_path,
                        verbose="ERROR",
                    )

                    window_starts = _ica_sources_diagnostic_window_starts(
                        raw,
                        settings=settings,
                    )

                    for window_number, start_s in enumerate(
                        window_starts,
                        start=1,
                    ):
                        _ica_sources_diagnostic_save_browser_page(
                            pdf,
                            raw=raw,
                            ica=ica,
                            sub=sub,
                            session=session,
                            start_s=float(start_s),
                            settings=settings,
                            page_number=window_number,
                            n_windows=len(window_starts),
                        )
                        wrote_any_page = True

                    _ica_sources_diagnostic_write_psd_pages(
                        pdf,
                        raw=raw,
                        ica=ica,
                        sub=sub,
                        session=session,
                        settings=settings,
                    )
                    wrote_any_page = True

                except Exception as error:
                    error_fig = plt.figure(figsize=(11, 8.5))

                    error_fig.text(
                        0.5,
                        0.63,
                        "ICA source diagnostic failed",
                        ha="center",
                        va="center",
                        fontsize=18,
                        fontweight="bold",
                        color="crimson",
                    )

                    error_fig.text(
                        0.5,
                        0.52,
                        f"Subject: {sub}\nSession: {session}",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

                    error_fig.text(
                        0.5,
                        0.34,
                        f"{type(error).__name__}: {error}\n\n"
                        f"Raw path:\n{raw_path}\n\n"
                        f"ICA path:\n{ica_path}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        wrap=True,
                    )

                    pdf.savefig(error_fig)
                    plt.close(error_fig)
                    wrote_any_page = True

                finally:
                    if raw is not None:
                        raw.close()

            if not found_input_pair:
                missing_fig = plt.figure(figsize=(11, 8.5))

                missing_fig.text(
                    0.5,
                    0.60,
                    f"No ICA raw/solution pairs found for {sub}",
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                )

                missing_fig.text(
                    0.5,
                    0.43,
                    f"Requested sessions: {', '.join(sessions)}",
                    ha="center",
                    va="center",
                    fontsize=11,
                )

                pdf.savefig(missing_fig)
                plt.close(missing_fig)

    return output_path


################################################################################
# Leadfield QC
################################################################################

@dataclass(frozen=True)
class NLGCLeadfieldQCSettings:
    """File naming, NLGC leadfield preparation, and report layout settings."""

    # Filename identifiers, without brackets.
    band_label: str = "1-10Hz"
    target_src_label: str = "vol18"
    forward_src_label: str = "vol5"

    # Display-only resampling of trial evoked data.
    display_sfreq: float = 25.0

    # NLGC leadfield preprocessing arguments.
    n_eigenmodes: int = 1
    n_orients: int = 2
    loose: float = 0.0
    depth: float = 0.0
    pca: bool = True

    # Set this only if your NLGC prepare_eigenmodes call requires an explicit
    # rank. If None, rank="info" is passed through to prepare_eigenmodes.
    rank: object = "info"

    # If True, subtract len(ica.exclude) from the computed MEG rank,
    # reproducing the rank handling in the supplied code.
    subtract_excluded_ica_components: bool = True

    # Leadfield displays can be extremely large. This limits the display
    # resolution only, not the preprocessing calculation.
    max_leadfield_columns_display: int = 1000
    max_gram_columns_display: int = 500
    max_whitener_columns_display: int = 400
    max_whitened_timepoints_display: int = 1000

    # PDF style.
    figsize: tuple[float, float] = (17, 11)
    cmap_signed: str = "RdBu_r"
    cmap_abs: str = "magma"
    cmap_data: str = "seismic"

def _nlgc_qc_evoked_path(
    sub: str,
    session: str,
    trial: int,
    *,
    megdir: pathlib.Path,
    settings: NLGCLeadfieldQCSettings,
) -> pathlib.Path:
    """Path to a processed trial evoked file."""
    return megdir / sub / (
        f"{sub}_{session}"
        f"-[trial={trial}]"
        f"-[{settings.band_label}]"
        f"-evoked-ave.fif"
    )


def _nlgc_qc_cov_path(
    sub: str,
    *,
    megdir: pathlib.Path,
    settings: NLGCLeadfieldQCSettings,
) -> pathlib.Path:
    """Path to the subject-level noise covariance."""
    return megdir / sub / (
        f"{sub}-[{settings.band_label}]-cov.fif"
    )


def _nlgc_qc_target_src_path(
    sub: str,
    *,
    mridir: pathlib.Path,
    settings: NLGCLeadfieldQCSettings,
) -> pathlib.Path:
    """Path to NLGC target source space."""
    return mridir / sub / "bem" / (
        f"{sub}-{settings.target_src_label}-src.fif"
    )


def _nlgc_qc_forward_path(
    sub: str,
    session: str,
    *,
    mridir: pathlib.Path,
    settings: NLGCLeadfieldQCSettings,
) -> pathlib.Path:
    """Path to the forward solution used for leadfield reduction."""
    return mridir / sub / "bem" / (
        f"{sub}_{session}"
        f"-[src={settings.forward_src_label}]"
        f"-solution-fwd.fif"
    )


def _nlgc_qc_ica_path(
    sub: str,
    session: str,
    *,
    megdir: pathlib.Path,
    wideband_label: str,
) -> pathlib.Path:
    """
    Path to the saved ICA solution.

    `wideband_label` is passed separately because your ICA filename uses
    a different band identifier, e.g. "1-100", than the evoked/covariance
    filename band, e.g. "1-10Hz".
    """
    return megdir / sub / (
        f"{sub}_tsss-{wideband_label}-{session}"
        f"-apply-comp-ica.fif"
    )

def _nlgc_qc_adjusted_rank(
    evoked: mne.Evoked,
    ica: Optional[mne.preprocessing.ICA],
    *,
    subtract_excluded: bool,
) -> tuple[dict, dict]:
    """
    Compute rank from evoked info and optionally subtract ICA exclusions.

    Returns
    -------
    raw_rank
        Rank reported by MNE before any ICA-exclusion adjustment.

    adjusted_rank
        Rank dictionary passed to leadfield preparation.
    """
    raw_rank = mne.compute_rank(
        evoked,
        rank="info",
        verbose="ERROR",
    )
    adjusted_rank = dict(raw_rank)

    if not subtract_excluded or ica is None:
        return raw_rank, adjusted_rank

    n_excluded = len(ica.exclude)

    # Your original code expects a magnetometer-only key. This version
    # handles common MNE rank-key variants more defensively.
    meg_keys = [
        key
        for key in adjusted_rank
        if key in {"meg", "mag", "grad"}
    ]

    if len(meg_keys) == 1:
        meg_key = meg_keys[0]
        adjusted_rank[meg_key] = max(
            1,
            adjusted_rank[meg_key] - n_excluded,
        )

    elif "meg" in adjusted_rank:
        adjusted_rank["meg"] = max(
            1,
            adjusted_rank["meg"] - n_excluded,
        )

    return raw_rank, adjusted_rank


def _nlgc_qc_even_indices(
    n_items: int,
    max_items: int,
) -> np.ndarray:
    """Choose evenly distributed indices for a bounded-resolution display."""
    if n_items <= 0:
        return np.array([], dtype=int)

    if n_items <= max_items:
        return np.arange(n_items, dtype=int)

    return np.unique(
        np.linspace(
            0,
            n_items - 1,
            num=max_items,
            dtype=int,
        )
    )


def _nlgc_qc_safe_symmetric_limit(
    array: np.ndarray,
    *,
    percentile: float = 99.0,
) -> float:
    """Choose a robust symmetric color range for signed matrices."""
    finite_values = np.abs(array[np.isfinite(array)])

    if finite_values.size == 0:
        return 1.0

    limit = float(np.percentile(finite_values, percentile))

    return max(limit, np.finfo(float).eps)


def _nlgc_qc_source_count(
    src_target: mne.SourceSpaces,
) -> int:
    """Count active vertices across all entries in a target source space."""
    return int(
        sum(
            np.sum(source_space["inuse"].astype(bool))
            for source_space in src_target
        )
    )

def _nlgc_qc_prepare_trial(
    *,
    evoked: mne.Evoked,
    cov: mne.Covariance,
    forward: mne.Forward,
    src_target: mne.SourceSpaces,
    ica: Optional[mne.preprocessing.ICA],
    settings: NLGCLeadfieldQCSettings,
) -> dict:
    """
    Recreate NLGC whitening and reduced-leadfield preprocessing for one trial.

    Important coordinate spaces
    ---------------------------
    Physical sensor space:
        M_raw has one row per selected channel in gain_info["ch_names"].

    Whitened/rank-reduced space:
        W may be rectangular with shape (effective_rank, n_sensors).
        Therefore M_whitened and G have `effective_rank` rows, not
        necessarily one row per physical sensor.
    """
    if src_target[0]["coord_frame"] != mne.io.constants.FIFF.FIFFV_COORD_HEAD:
        raise ValueError(
            "Target source space must be in head coordinates."
        )

    raw_rank, adjusted_rank = _nlgc_qc_adjusted_rank(
        evoked,
        ica,
        subtract_excluded=settings.subtract_excluded_ica_components,
    )

    prepare_rank = (
        adjusted_rank
        if settings.rank == "info"
        else settings.rank
    )['mag']

    (
        weights,
        G,
        label_vertidx,
        label_names,
        gain_info,
        whitener,
        singular_values,
    ) = prepare_eigenmodes(
        evoked.info,
        forward,
        cov,
        src_target,
        n_eigenmodes=settings.n_eigenmodes,
        n_orients=settings.n_orients,
        loose=settings.loose,
        depth=settings.depth,
        pca=settings.pca,
        rank=prepare_rank,
    )

    gain_channel_names = list(gain_info["ch_names"])

    missing_from_evoked = [
        channel_name
        for channel_name in gain_channel_names
        if channel_name not in evoked.ch_names
    ]

    if missing_from_evoked:
        raise RuntimeError(
            "Channels in gain_info['ch_names'] are absent from the Evoked "
            f"object: {missing_from_evoked}"
        )

    # These indices exist only in full evoked.data coordinates.
    evoked_row_indices = np.asarray(
        [
            evoked.ch_names.index(channel_name)
            for channel_name in gain_channel_names
        ],
        dtype=int,
    )

    M_raw = evoked.data[evoked_row_indices]
    M_whitened = whitener @ M_raw

    n_sensor_channels = len(gain_channel_names)
    n_whitened_modes = whitener.shape[0]

    # Physical-sensor-space checks.
    if M_raw.shape[0] != n_sensor_channels:
        raise RuntimeError(
            "M_raw row count does not match selected physical sensor count: "
            f"M_raw={M_raw.shape}, gain channels={n_sensor_channels}."
        )

    if whitener.shape[1] != n_sensor_channels:
        raise RuntimeError(
            "Whitener input dimension does not match selected physical "
            "sensor count: "
            f"whitener={whitener.shape}, "
            f"gain channels={n_sensor_channels}."
        )

    # Whitened rank-space checks.
    if M_whitened.shape[0] != n_whitened_modes:
        raise RuntimeError(
            "Whitened data row count does not match whitener output "
            f"dimension: M_whitened={M_whitened.shape}, "
            f"whitener={whitener.shape}."
        )

    if G.shape[0] != n_whitened_modes:
        raise RuntimeError(
            "Reduced leadfield row count does not match whitener output "
            f"dimension: G={G.shape}, whitener={whitener.shape}."
        )

    k_per_node = (
        settings.n_eigenmodes
        * settings.n_orients
    )

    if G.shape[1] % k_per_node != 0:
        raise RuntimeError(
            "Reduced leadfield columns are not divisible by K="
            f"{k_per_node}; G shape={G.shape}."
        )

    return {
        "weights": weights,
        "G": G,
        "label_vertidx": label_vertidx,
        "label_names": label_names,
        "gain_info": gain_info,
        "gain_channel_names": gain_channel_names,
        "evoked_row_indices": evoked_row_indices,
        "whitener": whitener,
        "singular_values": singular_values,
        "M_raw": M_raw,
        "M_whitened": M_whitened,
        "raw_rank": raw_rank,
        "adjusted_rank": adjusted_rank,
        "n_sensor_channels": n_sensor_channels,
        "n_whitened_modes": n_whitened_modes,
        "k_per_node": k_per_node,
    }


def _nlgc_qc_write_leadfield_pages(
    pdf: PdfPages,
    *,
    sub: str,
    session: str,
    trial: int,
    prepared: dict,
    settings: NLGCLeadfieldQCSettings,
) -> None:
    """
    Write reduced-leadfield, Gramian, overlap, and nodal-redundancy plots.

    Display geometry:
    - G is rectangular and uses aspect="auto".
    - Gramian, absolute overlap, and nodal redundancy are square matrices.
      They use aspect="equal" and a square axes box.
    - Matrix orientation matches the original diagnostic implementation:
      no transpose and no origin="lower".
    """
    G = prepared["G"]
    k_per_node = prepared["k_per_node"]

    if G.ndim != 2:
        raise ValueError(
            f"Expected a 2-D reduced leadfield, got G.shape={G.shape}."
        )

    if k_per_node <= 0:
        raise ValueError(
            f"k_per_node must be positive, got {k_per_node}."
        )

    if G.shape[1] % k_per_node != 0:
        raise RuntimeError(
            "Reduced leadfield columns are not divisible by the expected "
            f"K={k_per_node}: G.shape={G.shape}."
        )

    leadfield_columns = _nlgc_qc_even_indices(
        G.shape[1],
        settings.max_leadfield_columns_display,
    )
    gram_columns = _nlgc_qc_even_indices(
        G.shape[1],
        settings.max_gram_columns_display,
    )

    G_display = G[:, leadfield_columns]
    G_gram_display = G[:, gram_columns]

    # Match the original diagnostic:
    # S = F.T @ F
    gram = G_gram_display.T @ G_gram_display

    gram_limit = _nlgc_qc_safe_symmetric_limit(gram)
    abs_gram_limit = max(
        float(np.percentile(np.abs(gram), 99.0)),
        np.finfo(float).eps,
    )

    # Compute nodal redundancy on full G, not the display subset.
    #
    # For node r, sum the squared Frobenius norm of the off-diagonal
    # K x K Gramian blocks connecting node r to every other node.
    full_gram = G.T @ G
    n_nodes = G.shape[1] // k_per_node
    redundancy = np.zeros(n_nodes, dtype=float)

    for node_idx in range(n_nodes):
        row_start = node_idx * k_per_node
        row_stop = row_start + k_per_node

        for other_idx in range(n_nodes):
            if other_idx == node_idx:
                continue

            col_start = other_idx * k_per_node
            col_stop = col_start + k_per_node

            block = full_gram[
                row_start:row_stop,
                col_start:col_stop,
            ]

            redundancy[node_idx] += (
                np.linalg.norm(block, ord="fro") ** 2
            )

    redundancy_outer = np.outer(
        redundancy,
        redundancy,
    )

    fig = plt.figure(figsize=settings.figsize)

    grid = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        left=0.06,
        right=0.94,
        bottom=0.09,
        top=0.90,
        hspace=0.34,
        wspace=0.28,
    )

    ax_leadfield = fig.add_subplot(grid[0, 0])
    ax_gram = fig.add_subplot(grid[0, 1])
    ax_overlap = fig.add_subplot(grid[1, 0])
    ax_redundancy = fig.add_subplot(grid[1, 1])

    # G is inherently rectangular:
    # whitened sensor/rank modes x reduced source modes.
    image = ax_leadfield.imshow(
        G_display,
        aspect="auto",
        cmap=settings.cmap_data,
        interpolation="nearest",
    )
    ax_leadfield.set_title("Reduced leadfield: G")
    ax_leadfield.set_xlabel("Displayed reduced source-mode index")
    ax_leadfield.set_ylabel("Whitened sensor-mode index")
    fig.colorbar(
        image,
        ax=ax_leadfield,
        pad=0.01,
        label="Leadfield weight",
    )

    # Source-mode x source-mode; preserve original array orientation.
    image = ax_gram.imshow(
        gram,
        aspect="equal",
        cmap=settings.cmap_signed,
        vmin=-gram_limit,
        vmax=gram_limit,
        interpolation="nearest",
    )
    ax_gram.set_box_aspect(1)
    ax_gram.set_title(
        r"Signed Gramian: $G^\top G$"
    )
    ax_gram.set_xlabel("Displayed source-mode index")
    ax_gram.set_ylabel("Displayed source-mode index")
    fig.colorbar(
        image,
        ax=ax_gram,
        pad=0.01,
        label="Signed sensor-mode inner product",
    )

    # Source-mode x source-mode; preserve original array orientation.
    image = ax_overlap.imshow(
        np.abs(gram),
        aspect="equal",
        cmap=settings.cmap_abs,
        vmin=0,
        vmax=abs_gram_limit,
        interpolation="nearest",
    )
    ax_overlap.set_box_aspect(1)
    ax_overlap.set_title(
        r"Absolute overlap: $|G^\top G|$"
    )
    ax_overlap.set_xlabel("Displayed source-mode index")
    ax_overlap.set_ylabel("Displayed source-mode index")
    fig.colorbar(
        image,
        ax=ax_overlap,
        pad=0.01,
        label="Absolute sensor-mode overlap",
    )

    # Node x node; preserve original outer-product orientation.
    image = ax_redundancy.imshow(
        redundancy_outer,
        aspect="equal",
        cmap=settings.cmap_abs,
        interpolation="nearest",
    )
    ax_redundancy.set_box_aspect(1)
    ax_redundancy.set_title(
        "Nodal redundancy outer product"
    )
    ax_redundancy.set_xlabel("Source node")
    ax_redundancy.set_ylabel("Source node")
    fig.colorbar(
        image,
        ax=ax_redundancy,
        pad=0.01,
        label="Nodal redundancy product",
    )

    fig.suptitle(
        f"Reduced leadfield diagnostic | Subject: {sub} | "
        f"Session: {session} | Trial: {trial}",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    fig.text(
        0.5,
        0.018,
        f"Full reduced leadfield: G={G.shape} | "
        f"K={k_per_node} retained columns per patch/node | "
        f"Reduced patches/nodes={n_nodes}. "
        f"Gramian and overlap use {len(gram_columns)}/{G.shape[1]} "
        f"displayed source modes; nodal redundancy is computed from "
        "the full reduced leadfield.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    pdf.savefig(fig)
    plt.close(fig)


def _nlgc_qc_write_singular_values_page(
    pdf: PdfPages,
    *,
    sub: str,
    session: str,
    trial: int,
    prepared: dict,
    settings: NLGCLeadfieldQCSettings,
) -> None:
    """
    Plot retained singular values and patch-level leadfield energy.

    Diagnostic panels
    -----------------
    1. Retained singular values by patch/node and retained mode.
    2. Within-patch mode-energy shares.
    3. Total retained squared-singular-value energy per patch:
           E_p = sum_k sigma_(p,k)^2
    4. Ratio sigma_1 / sigma_last for every patch.

    Cumulative energy is deliberately omitted because it always equals one
    at the last retained component and is therefore uninformative with a
    small number of retained modes.
    """
    singular_values = prepared["singular_values"]

    try:
        singular_array = np.asarray(
            singular_values,
            dtype=float,
        )

        if singular_array.ndim == 1:
            singular_array = singular_array[np.newaxis, :]

        if singular_array.ndim != 2:
            raise ValueError(
                "Expected singular_values with one or two dimensions; "
                f"got {singular_array.shape}."
            )

        singular_array = np.abs(singular_array)

        n_nodes, n_modes = singular_array.shape

        singular_energy = singular_array ** 2
        total_patch_energy = singular_energy.sum(axis=1)

        mode_energy_fraction = singular_energy / np.maximum(
            total_patch_energy[:, np.newaxis],
            np.finfo(float).eps,
        )

        sigma_first = singular_array[:, 0]
        sigma_last = singular_array[:, -1]

        retained_condition_ratio = sigma_first / np.maximum(
            sigma_last,
            np.finfo(float).eps,
        )

        node_indices = np.arange(n_nodes)

    except Exception as error:
        fig = plt.figure(figsize=settings.figsize)

        fig.text(
            0.5,
            0.58,
            "Could not render singular-value diagnostics",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="crimson",
        )

        fig.text(
            0.5,
            0.40,
            f"{type(error).__name__}: {error}",
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
        )

        pdf.savefig(fig)
        plt.close(fig)
        return

    fig = plt.figure(figsize=settings.figsize)

    grid = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        left=0.07,
        right=0.95,
        bottom=0.10,
        top=0.90,
        hspace=0.36,
        wspace=0.28,
    )

    ax_values = fig.add_subplot(grid[0, 0])
    ax_mode_share = fig.add_subplot(grid[0, 1])
    ax_total_energy = fig.add_subplot(grid[1, 0])
    ax_condition = fig.add_subplot(grid[1, 1])

    image = ax_values.imshow(
        singular_array,
        aspect="auto",
        cmap=settings.cmap_abs,
        interpolation="nearest",
        origin="lower",
    )
    ax_values.set_title("Retained singular values")
    ax_values.set_xlabel("Retained mode index")
    ax_values.set_ylabel("Patch/node index")
    fig.colorbar(
        image,
        ax=ax_values,
        pad=0.01,
        label="Singular value",
    )

    image = ax_mode_share.imshow(
        mode_energy_fraction,
        aspect="auto",
        cmap=settings.cmap_abs,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        origin="lower",
    )
    ax_mode_share.set_title("Within-patch retained-mode energy share")
    ax_mode_share.set_xlabel("Retained mode index")
    ax_mode_share.set_ylabel("Patch/node index")
    fig.colorbar(
        image,
        ax=ax_mode_share,
        pad=0.01,
        label=r"$\sigma_k^2 / \sum_j \sigma_j^2$",
    )

    # New: absolute retained leadfield energy for every patch/node.
    # A log scale makes weakly represented patches visible alongside strong
    # patches while preserving relative differences.
    safe_patch_energy = np.maximum(
        total_patch_energy,
        np.finfo(float).tiny,
    )

    ax_total_energy.plot(
        node_indices,
        safe_patch_energy,
        color="crimson",
        linewidth=0.8,
    )
    ax_total_energy.set_yscale("log")
    ax_total_energy.set_title(
        r"Total retained variance by patch: $\sum_k \sigma_k^2$"
    )
    ax_total_energy.set_xlabel("Patch/node index")
    ax_total_energy.set_ylabel("Retained squared-SV energy")
    ax_total_energy.grid(
        color="0.88",
        linewidth=0.5,
    )

    ax_condition.plot(
        node_indices,
        retained_condition_ratio,
        color="black",
        linewidth=0.8,
    )
    ax_condition.set_yscale("log")
    ax_condition.set_title(
        r"Retained SV ratio: $\sigma_1 / \sigma_{\mathrm{last}}$"
    )
    ax_condition.set_xlabel("Patch/node index")
    ax_condition.set_ylabel("Condition ratio")
    ax_condition.grid(
        color="0.88",
        linewidth=0.5,
    )

    fig.suptitle(
        f"Leadfield reduction spectrum | Subject: {sub} | "
        f"Session: {session} | Trial: {trial}",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    energy_median = float(np.median(total_patch_energy))
    energy_min = float(np.min(total_patch_energy))
    energy_max = float(np.max(total_patch_energy))

    fig.text(
        0.5,
        0.020,
        f"Retained singular-value array: {singular_array.shape} | "
        r"Total patch energy is $\sum_k \sigma_k^2$ | "
        f"min={energy_min:.3e}, median={energy_median:.3e}, "
        f"max={energy_max:.3e}. "
        "The energy panel is log-scaled to make weak and strong patches "
        "simultaneously visible.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    pdf.savefig(fig)
    plt.close(fig)


def _nlgc_qc_write_summary_page(
    pdf: PdfPages,
    *,
    sub: str,
    session: str,
    trial: int,
    evoked: mne.Evoked,
    cov: mne.Covariance,
    forward: mne.Forward,
    src_target: mne.SourceSpaces,
    ica: Optional[mne.preprocessing.ICA],
    prepared: dict,
    settings: NLGCLeadfieldQCSettings,
) -> None:
    """Write a descriptive metadata page for one NLGC diagnostic trial."""
    G = prepared["G"]
    whitener = prepared["whitener"]
    M_raw = prepared["M_raw"]
    M_whitened = prepared["M_whitened"]

    fig = plt.figure(figsize=settings.figsize)
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])
    ax.set_axis_off()

    raw_rank = prepared["raw_rank"]
    adjusted_rank = prepared["adjusted_rank"]

    lines = [
        "NLGC Whitener and Leadfield Diagnostic",
        "",
        f"Subject: {sub}",
        f"Session: {session}",
        f"Trial: {trial}",
        "",
        "Input data",
        f"Evoked channels: {len(evoked.ch_names)}",
        f"Evoked sampling frequency: {evoked.info['sfreq']:.3f} Hz",
        f"Evoked time range: {evoked.times[0]:.3f} to "
        f"{evoked.times[-1]:.3f} s",
        f"Noise covariance channels: {len(cov['names'])}",
        f"Forward rows / columns: {forward['sol']['data'].shape}",
        f"Active target-source vertices: "
        f"{_nlgc_qc_source_count(src_target)}",
        "",
        "ICA / rank treatment",
        f"ICA exclusions: {list(ica.exclude) if ica is not None else 'ICA not loaded'}",
        f"MNE rank before ICA adjustment: {raw_rank}",
        f"Rank supplied to prepare_eigenmodes: {adjusted_rank}",
        "",
        "NLGC leadfield preparation",
        f"Target source space label: {settings.target_src_label}",
        f"Forward source space label: {settings.forward_src_label}",
        f"Evoked/covariance band label: {settings.band_label}",
        f"Eigenmodes per node: {settings.n_eigenmodes}",
        f"Orientations per eigenmode: {settings.n_orients}",
        f"Columns per source node (K): {prepared['k_per_node']}",
        f"Reduced leadfield G shape: {G.shape}",
        f"Whitener shape: {whitener.shape}",
        f"Unwhitened selected data M_raw shape: {M_raw.shape}",
        f"Whitened data M shape: {M_whitened.shape}",
        f"Reduced source nodes: {G.shape[1] // prepared['k_per_node']}",
    ]

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
    )

    pdf.savefig(fig)
    plt.close(fig)

def _nlgc_qc_write_whitener_page(
    pdf: PdfPages,
    *,
    sub: str,
    session: str,
    trial: int,
    prepared: dict,
    settings: NLGCLeadfieldQCSettings,
) -> None:
    """
    Plot raw and whitened data in equal-sized panels plus the whitener.

    M_raw:
        physical sensors × time

    W:
        whitened modes × physical sensors

    M_whitened:
        whitened modes × time

    For readability, data are transposed in the image panels:
        rows    = time
        columns = sensor/mode

    Both data panels occupy equal GridSpec cells.
    """
    M_raw = prepared["M_raw"]
    M_whitened = prepared["M_whitened"]
    whitener = prepared["whitener"]

    n_sensor_channels = prepared["n_sensor_channels"]
    n_whitened_modes = prepared["n_whitened_modes"]

    if M_raw.shape[0] != n_sensor_channels:
        raise RuntimeError(
            f"M_raw={M_raw.shape}, expected {n_sensor_channels} "
            "physical-sensor rows."
        )

    if whitener.shape != (
        n_whitened_modes,
        n_sensor_channels,
    ):
        raise RuntimeError(
            f"Whitener={whitener.shape}, expected "
            f"({n_whitened_modes}, {n_sensor_channels})."
        )

    if M_whitened.shape[0] != n_whitened_modes:
        raise RuntimeError(
            f"M_whitened={M_whitened.shape}, expected "
            f"{n_whitened_modes} whitened-mode rows."
        )

    sensor_indices = _nlgc_qc_even_indices(
        n_sensor_channels,
        settings.max_whitener_columns_display,
    )
    whitened_indices = _nlgc_qc_even_indices(
        n_whitened_modes,
        settings.max_whitener_columns_display,
    )
    time_indices = _nlgc_qc_even_indices(
        M_raw.shape[1],
        settings.max_whitened_timepoints_display,
    )

    raw_display = M_raw[
        np.ix_(sensor_indices, time_indices)
    ]
    whitened_display = M_whitened[
        np.ix_(whitened_indices, time_indices)
    ]
    whitener_display = whitener[
        np.ix_(whitened_indices, sensor_indices)
    ]

    raw_limit = _nlgc_qc_safe_symmetric_limit(raw_display)
    whitened_limit = _nlgc_qc_safe_symmetric_limit(
        whitened_display
    )
    whitener_limit = _nlgc_qc_safe_symmetric_limit(
        whitener_display
    )

    fig = plt.figure(figsize=settings.figsize)

    grid = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        left=0.07,
        right=0.94,
        bottom=0.09,
        top=0.90,
        hspace=0.36,
        wspace=0.30,
        height_ratios=(1.0, 1.0),
        width_ratios=(1.0, 1.0),
    )

    ax_raw = fig.add_subplot(grid[0, 0])
    ax_whitened = fig.add_subplot(grid[0, 1])
    ax_whitener = fig.add_subplot(grid[1, :])

    # Equal-sized top-left panel.
    image = ax_raw.imshow(
        raw_display.T,
        aspect="auto",
        cmap=settings.cmap_data,
        vmin=-raw_limit,
        vmax=raw_limit,
        interpolation="nearest",
        origin="lower",
    )
    ax_raw.set_title(
        r"Selected evoked data: $M_{\mathrm{raw}}$",
        fontsize=12,
    )
    ax_raw.set_xlabel("Physical sensor index")
    ax_raw.set_ylabel("Display time index")
    fig.colorbar(
        image,
        ax=ax_raw,
        pad=0.01,
        label="Sensor amplitude",
    )

    # Same GridSpec-cell dimensions as M_raw.
    image = ax_whitened.imshow(
        whitened_display.T,
        aspect="auto",
        cmap=settings.cmap_data,
        vmin=-whitened_limit,
        vmax=whitened_limit,
        interpolation="nearest",
        origin="lower",
    )
    ax_whitened.set_title(
        r"Whitened data: $W M_{\mathrm{raw}}$",
        fontsize=12,
    )
    ax_whitened.set_xlabel("Whitened mode index")
    ax_whitened.set_ylabel("Display time index")
    fig.colorbar(
        image,
        ax=ax_whitened,
        pad=0.01,
        label="Whitened amplitude",
    )

    # The whitener has an intrinsic near-square shape, e.g. 144 × 147.
    # `aspect="equal"` retains correct numerical geometry rather than
    # stretching it across the complete lower-row rectangle.
    image = ax_whitener.imshow(
        whitener_display,
        aspect="equal",
        cmap=settings.cmap_data,
        vmin=-whitener_limit,
        vmax=whitener_limit,
        interpolation="nearest",
        origin="lower",
    )
    ax_whitener.set_title(
        r"Rank-reducing whitener: $W$",
        fontsize=12,
    )
    ax_whitener.set_xlabel("Physical sensor index")
    ax_whitener.set_ylabel("Whitened mode index")
    fig.colorbar(
        image,
        ax=ax_whitener,
        pad=0.015,
        label="Whitening weight",
    )

    fig.suptitle(
        f"Whitening diagnostic | Subject: {sub} | "
        f"Session: {session} | Trial: {trial}",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    fig.text(
        0.5,
        0.018,
        f"Physical sensors: {n_sensor_channels} | "
        f"Whitened rank: {n_whitened_modes} | "
        f"$M_{{raw}}$: {M_raw.shape} | "
        f"$W$: {whitener.shape} | "
        f"$WM_{{raw}}$: {M_whitened.shape}. "
        "The two data images have equal panel dimensions; data are shown "
        "as time × channel/mode.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    pdf.savefig(fig)
    plt.close(fig)
    

def make_nlgc_whitener_leadfield_diagnostic_pdf(
    subjects: Iterable[str],
    sessions: Iterable[str],
    trials: Iterable[int],
    config,
    *,
    megdir: str | pathlib.Path,
    mridir: str | pathlib.Path,
    outdir: str | pathlib.Path,
    
    settings: Optional[NLGCLeadfieldQCSettings] = None,
    output_name: str = "nlgc_whitener_leadfield_diagnostics.pdf",
    overwrite: bool = True,
) -> pathlib.Path:
    """
    Create a multipage NLGC whitener/leadfield diagnostic PDF.

    The function does not use the pipeline config object. File-identifying
    values are supplied directly through `settings` and `wideband_label`.

    For each available subject/session/trial evoked file, it loads:
    - processed trial Evoked;
    - subject-level covariance;
    - target source space;
    - session-level forward model;
    - session-level ICA solution, when available.

    It then runs `prepare_eigenmodes()` with the requested settings and writes:
    1. A metadata and shape/rank summary page.
    2. Raw selected data, whitener, and whitened data.
    3. Reduced leadfield, signed/absolute Gramian, and nodal redundancy.
    4. Singular-value diagnostics, when available.

    Missing trial Evoked files are skipped. Missing subject-level prerequisites
    produce an explicit error page for that subject/session/trial rather than
    terminating the full PDF.
    """
    if settings is None:
        lb_analysis = config.filter_params.analysis_lower_bandlimit
        ub_analysis = config.filter_params.analysis_upper_bandlimit
        lb_wideband = config.filter_params.wideband_lower_bandlimit
        ub_wideband = config.filter_params.wideband_upper_bandlimit
        wideband_label = f"{lb_wideband}-{ub_wideband}"
        settings = NLGCLeadfieldQCSettings(
            band_label=f"{lb_analysis}-{ub_analysis}Hz"
        )

    subjects = list(subjects)
    sessions = list(sessions)
    trials = list(trials)

    if not subjects:
        raise ValueError("subjects cannot be empty.")
    if not sessions:
        raise ValueError("sessions cannot be empty.")
    if not trials:
        raise ValueError("trials cannot be empty.")

    megdir = pathlib.Path(megdir)
    mridir = pathlib.Path(mridir)
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_path = outdir / output_name

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists and overwrite=False: {output_path}"
        )

    with PdfPages(output_path) as pdf:
        for sub in subjects:
            for session in sessions:
                cov_path = _nlgc_qc_cov_path(
                    sub,
                    megdir=megdir,
                    settings=settings,
                )
                src_target_path = _nlgc_qc_target_src_path(
                    sub,
                    mridir=mridir,
                    settings=settings,
                )
                forward_path = _nlgc_qc_forward_path(
                    sub,
                    session,
                    mridir=mridir,
                    settings=settings,
                )
                ica_path = _nlgc_qc_ica_path(
                    sub,
                    session,
                    megdir=megdir,
                    wideband_label=wideband_label,
                )

                for trial in trials:
                    evoked_path = _nlgc_qc_evoked_path(
                        sub,
                        session,
                        trial,
                        megdir=megdir,
                        settings=settings,
                    )

                    # Requested behavior: missing trial data are skipped.
                    if not evoked_path.exists():
                        continue

                    prerequisite_paths = {
                        "Noise covariance": cov_path,
                        "Target source space": src_target_path,
                        "Forward solution": forward_path,
                    }

                    missing_prerequisites = {
                        label: path
                        for label, path in prerequisite_paths.items()
                        if not path.exists()
                    }

                    if missing_prerequisites:
                        fig = plt.figure(figsize=(11, 8.5))

                        fig.text(
                            0.5,
                            0.65,
                            "NLGC diagnostic prerequisites missing",
                            ha="center",
                            va="center",
                            fontsize=18,
                            fontweight="bold",
                            color="crimson",
                        )

                        fig.text(
                            0.5,
                            0.53,
                            f"Subject: {sub}\n"
                            f"Session: {session}\n"
                            f"Trial: {trial}",
                            ha="center",
                            va="center",
                            fontsize=12,
                        )

                        missing_text = "\n\n".join(
                            f"{label}:\n{path}"
                            for label, path
                            in missing_prerequisites.items()
                        )

                        fig.text(
                            0.5,
                            0.30,
                            missing_text,
                            ha="center",
                            va="center",
                            fontsize=9,
                            wrap=True,
                        )

                        pdf.savefig(fig)
                        plt.close(fig)
                        continue

                    try:
                        evoked = mne.read_evokeds(
                            evoked_path,
                            condition=0,
                            verbose="ERROR",
                        )

                        # Display resampling mirrors your NLGC diagnostic
                        # example. It happens before prepare_eigenmodes.
                        if evoked.info["sfreq"] > settings.display_sfreq:
                            evoked.resample(
                                settings.display_sfreq,
                            )

                        cov = mne.read_cov(
                            cov_path,
                            verbose="ERROR",
                        )

                        src_target = mne.read_source_spaces(
                            src_target_path,
                            verbose="ERROR",
                        )

                        forward = mne.read_forward_solution(
                            forward_path,
                            verbose="ERROR",
                        )

                        # ICA is needed only for optional rank adjustment.
                        # A missing ICA sidecar does not block the remaining
                        # whitener/leadfield diagnostic.
                        ica = None
                        if ica_path.exists():
                            ica = mne.preprocessing.read_ica(
                                ica_path,
                                verbose="ERROR",
                            )

                        prepared = _nlgc_qc_prepare_trial(
                            evoked=evoked,
                            cov=cov,
                            forward=forward,
                            src_target=src_target,
                            ica=ica,
                            settings=settings,
                        )

                        _nlgc_qc_write_summary_page(
                            pdf,
                            sub=sub,
                            session=session,
                            trial=trial,
                            evoked=evoked,
                            cov=cov,
                            forward=forward,
                            src_target=src_target,
                            ica=ica,
                            prepared=prepared,
                            settings=settings,
                        )

                        _nlgc_qc_write_whitener_page(
                            pdf,
                            sub=sub,
                            session=session,
                            trial=trial,
                            prepared=prepared,
                            settings=settings,
                        )

                        _nlgc_qc_write_leadfield_pages(
                            pdf,
                            sub=sub,
                            session=session,
                            trial=trial,
                            prepared=prepared,
                            settings=settings,
                        )

                        _nlgc_qc_write_singular_values_page(
                            pdf,
                            sub=sub,
                            session=session,
                            trial=trial,
                            prepared=prepared,
                            settings=settings,
                        )

                    except Exception as error:
                        fig = plt.figure(figsize=(11, 8.5))

                        fig.text(
                            0.5,
                            0.65,
                            "NLGC whitener / leadfield diagnostic failed",
                            ha="center",
                            va="center",
                            fontsize=18,
                            fontweight="bold",
                            color="crimson",
                        )

                        fig.text(
                            0.5,
                            0.53,
                            f"Subject: {sub}\n"
                            f"Session: {session}\n"
                            f"Trial: {trial}",
                            ha="center",
                            va="center",
                            fontsize=12,
                        )

                        fig.text(
                            0.5,
                            0.31,
                            f"{type(error).__name__}: {error}\n\n"
                            f"Evoked:\n{evoked_path}\n\n"
                            f"Covariance:\n{cov_path}\n\n"
                            f"Target source space:\n{src_target_path}\n\n"
                            f"Forward:\n{forward_path}",
                            ha="center",
                            va="center",
                            fontsize=8.5,
                            wrap=True,
                        )

                        pdf.savefig(fig)
                        plt.close(fig)

    return output_path