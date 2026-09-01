from __future__ import annotations
import mne
import pathlib
from mne_icalabel import label_components
from nlgc_pipeline.utils.surfaces import make_bem
from nlgc_pipeline.utils.qc import (maxwell_flat_qc, score_ica_cardiac, 
                                    robust_noisy_meg_channels)
from nlgc_pipeline.utils.annotations import trial_relative_annotations
from nlgc_pipeline.utils.coreg import compute_coreg
import numpy as np


SKIP_ANNOTATIONS = ['edge', 
                    'BAD_ACQ_SKIP',
                    'BAD_GLOBAL_ARTIFACT']


def fit_filters(sub, config, verbose=False):
    assert config.data_src.megdir is not None, \
        "MEG directory has not been initialized in pipeline_config!"
    assert config.scan_info.session != 'UNK', \
        "Please initialize session in the configuration!"

    if (config.system_spec.meg_sys == 'KIT'):
        st_only=True
    elif (config.system_spec.meg_sys == 'MEGIN'):
        st_only=False
    else:
        raise RuntimeError("Invalid MEG System in config!")

    # TODO incorporate visit and trial info
    raw_path = pathlib.Path((f"{config.data_src.megdir}/{sub}/{sub}_"
                            f"{config.scan_info.session}-raw.fif"))
    
    assert raw_path.exists(), \
        f"Raw MEG file does not exist! Current file path: {raw_path}"

    megout, mriout = _verify_outdir(sub, config)
    
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    raw = raw.resample(config.filter_params.sfreq)

    # initial impression of channels that dominating high-variance noise
    bads_var, scores = robust_noisy_meg_channels(raw)
    raw.info['bads'] += bads_var

    if config.verbose:
        print(f"omitting high-variance channels {bads_var}")

    # auto coregister AND drop implausible dig points, which can be important
    # for maxwell fitering in later steps
    raw = compute_coreg(sub, config, raw, verbose)

    # In case we are working with MEGIN data, we need to get rid of projection
    # vectors before ICA; the SSP projection vectors are PCA components applied
    # during the acquisition process for on-the-fly signal quality assessment
    # and are generally not used for actual pre-processing, especially if
    # maxwell and ICA are to be applied to the data. (Generally, these
    # projectors will be set to 'active' = False, meaning that MNE will not
    # apply them anyway.) These should be zeroed before any preprocessing is
    # done so that the projectors are not applied during whitening.
    raw.del_proj() 

    # bad channel identification
    if config.verbose:
        print("identifying bad channels")
    raw_qc = raw.pick(picks='meg')
    
    auto_noisy_chs, auto_flat_chs, auto_scores = \
            mne.preprocessing.find_bad_channels_maxwell(raw_qc, 
                                            coord_frame='head', 
                                            ignore_ref=True, 
                                            return_scores=True,
                                            skip_by_annotation=SKIP_ANNOTATIONS,
                                            verbose=False)
    
    # Raw maxwell filtering can be too aggressive when applied over entire
    # recording (subject movement during breaks between trials can often cause
    # clipping). This function 1) identifies those data segments and annotates
    # them so that they aren't used during downstream filter fitting; and 2)
    # identifies truly flat channels that are distinct from those with
    # aforementioned transients
    persistent_flat_chs, flat_annotations, flat_qc = maxwell_flat_qc(
        auto_scores,
        auto_flat_chs=auto_flat_chs,
        global_flat_fraction=0.25,
        persistent_flat_fraction=0.80,
        min_persistent_bins=3,
        min_global_bins=1,
        pad_sec=0.25,
    )
    annots = raw.annotations + flat_annotations

    annots.save(fname=(
        f"{megout}/{sub}-"
        f"{config.scan_info.session}-annotations.csv"), 
        overwrite=config.data_src.overwrite
    )

    if config.verbose:
        print("Auto-flat anywhere:", auto_flat_chs)
        print("Persistent flat outside global bad bins:", persistent_flat_chs)

        print("Global bad bins:")
        for bin_idx in np.flatnonzero(flat_qc["global_bin_mask"]):
            print(
                bin_idx,
                auto_scores["bins"][bin_idx],
                f"flat fraction={flat_qc['bin_flat_fraction'][bin_idx]:.3f}",
            )

    raw.info['bads'] += auto_noisy_chs + persistent_flat_chs
    raw.set_annotations(annots)

    if config.verbose:
        print(f"{raw.info['bads']=}")

    # maxwell filter
    if config.verbose:
        print("maxwell filtering")
    raw_tsss = mne.preprocessing.maxwell_filter(raw, st_duration=10, 
                                            ignore_ref=True, 
                                            st_correlation=0.9, 
                                            st_only=st_only, 
                                            bad_condition='warning', 
                                            coord_frame='head', 
                                            skip_by_annotation=SKIP_ANNOTATIONS,
                                            verbose=False)
    
    # bandpass filter
    if config.verbose:
        print("wide band filtering")
    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit
    phase = config.filter_params.filter_phase
    tsss_causal = raw_tsss.filter(l_freq=l_filt, h_freq=h_filt, picks='meg', 
                                  phase=phase, verbose=verbose,
                                  skip_by_annotation=SKIP_ANNOTATIONS)
   
    tsss_causal.save(fname=(
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-raw.fif"), 
        overwrite=config.data_src.overwrite
    )

    # ica fit
    if config.verbose:
        print("fit ica")
    ica = mne.preprocessing.ICA(
        n_components=config.filter_params.ICA_components,
        method=config.filter_params.ICA_method,
    )
    
    ica.fit(tsss_causal, reject_by_annotation=True)

    ica.save((
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-ica.fif"), 
        overwrite=config.data_src.overwrite
    )

    return tsss_causal, ica


def apply_ica(sub, config, tsss_causal=None, ica=None, verbose=False):
    if config.verbose:
        print(f"apply ica with mode={config.filter_params.ICA_mode}")
    assert config.data_src.megdir is not None, \
        "MEG directory has not been initialized in pipeline_config!"
    assert config.scan_info.session != 'UNK', \
        "Please initialize session in the configuration!"

    megout, mriout = _verify_outdir(sub, config)
    
    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit

    tsss_path = pathlib.Path((
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-raw.fif"))
    
    ica_path = pathlib.Path((
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-ica.fif"))
    
    if tsss_causal is None:
        assert tsss_path.exists(), \
            "Must fit ICA before applying. Run fit_ica() before this function"
        tsss_causal = mne.io.read_raw_fif(tsss_path, preload=True)

    if ica is None:
        assert ica_path.exists(), \
            "Must fit ICA before applying. Run fit_ica() before this function"
        ica = mne.preprocessing.read_ica(ica_path)

    if (config.filter_params.ICA_mode=='manual'):
        return _manual_ica(sub, config, tsss_causal, ica, megout)
    elif(config.filter_params.ICA_mode=='auto'):
        return _auto_ica(sub, config, tsss_causal, ica, megout)
    elif(config.filter_params.ICA_mode=='auto-strict'):
        return _auto_ica(sub, config, tsss_causal, ica, megout, strict=True)
    else:
        raise RuntimeError(("config.ICA_mode set up incorrectly!"))


def _manual_ica(sub, config, tsss_causal, ica, megout):
    ica.plot_components(list(range(20)))
    ica.plot_sources(tsss_causal, block=True)

    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit

    ica.save((
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-apply-comp-ica.fif"), 
        overwrite=config.data_src.overwrite
    )
    ica_apply = ica.apply(tsss_causal.copy(), exclude=ica.exclude)

    ica_apply.save((
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-ica-apply-raw.fif"), 
        overwrite=config.data_src.overwrite
    )
    return ica_apply


def _auto_ica(sub, config, tsss_causal, ica, megout, strict=False):
    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit
    
    labels = label_components(tsss_causal, ica, method='megnet')
    print(labels)
    for idx, label in enumerate(labels['labels']):
        # exclude any non-brain activity
        if strict:
            if "brain" not in label:
                ica.exclude.append(idx)

        # only exclude blinks and heartbeats
        else:
            if label in ['eye blink', 'heart beat']:
                ica.exclude.append(idx)

    ica_apply = ica.apply(tsss_causal.copy(), exclude=ica.exclude)

    # megnet can miss heartbeats if they explain low variance in the sensors but
    # we want to be sure to remove them to avoid systematic shocks to the kf.
    # find_ecg_events works without reference channels and can pick heartbeat
    # events using templating
    ecg_events, _, pulse, ecg = mne.preprocessing.find_ecg_events(
        tsss_causal,
        ch_name=None,
        event_id=999,
        reject_by_annotation=True,
        return_ecg=True,
        verbose=True,
    )

    # project the ecg events onto the ica componets and remove ones that are
    # highly correlated
    cardiac_scores, cardiac_qc = score_ica_cardiac(
        tsss_causal.copy().filter(5.0, 35.0), # filter to QRS waveform band
        ica,
        ecg_events=ecg_events,
        reject_threshold=0.20,  # reject by statistical threshold, OR
        min_r2_lock=0.02,       # correlation, OR
        min_split_r=0.80,       # split waveform correlation
    )

    cardiac_inds = cardiac_scores.index[
        cardiac_scores["reject"]
    ].tolist()

    if config.verbose:
        print(cardiac_scores)
        print("Automatic cardiac exclusions:", cardiac_inds)

    ica.exclude = sorted(set(ica.exclude).union(cardiac_inds))

    ica_apply.save((
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-ica-apply-raw.fif"), 
        overwrite=config.data_src.overwrite
    )
    
    ica.save((
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-apply-comp-ica.fif"), 
        overwrite=config.data_src.overwrite
    )
    
    print(type(ica_apply))
    return ica_apply


def filter_empty(sub, config, verbose=False):
    print("Filtering emptyroom...")
    assert config.data_src.megdir is not None, \
        "MEG directory has not been initialized in pipeline_config!"
    
    empty_raw_path = pathlib.Path(
        f"{config.data_src.megdir}/{sub}/{sub}_emptyroom-raw.fif"
    )

    assert empty_raw_path.exists(), "Raw emptyroom does not exist!"
    assert pathlib.Path((
            f"{config.data_src.megdir}/{sub}/{sub}_"
            f"{config.scan_info.session}-raw.fif")).exists(), \
        ("empty room processing requires info from the raw MEG file to "
         "be properly oriented! Raw file not found!")
    
    if (config.system_spec.meg_sys=='KIT'):
        st_only=True
    elif (config.system_spec.meg_sys=='MEGIN'):
        st_only=False
    else:
        raise RuntimeError("Invalid MEG System in config!")

    megout, mriout = _verify_outdir(sub, config)

    raw_empty = mne.io.read_raw_fif(empty_raw_path, preload=True)
    raw_empty.del_proj() # Necessary for some MEG Systems to be compatible
    
    raw_empty = raw_empty.pick(picks='meg', exclude='bads')
    
    raw_empty = raw_empty.resample(config.filter_params.sfreq)

    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit
    raw = mne.io.read_raw_fif((
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-raw.fif"),
        preload=False
    )

    # make sure bad channels are the same
    raw_empty.info["bads"] = raw.info["bads"].copy()

    # causes null space malalignment if not present for covariance
    raw_empty.info['dev_head_t'] = raw.info['dev_head_t'] 
    
    # maxwell filter
    empty_tsss = mne.preprocessing.maxwell_filter(raw_empty, st_duration=10,
                                            ignore_ref=True, 
                                            st_correlation=0.9, 
                                            st_only=st_only, 
                                            bad_condition='warning', 
                                            coord_frame='head', 
                                            skip_by_annotation=SKIP_ANNOTATIONS,
                                            verbose=verbose)
    
    # bandpass filter
    phase = config.filter_params.filter_phase
    empty_filtered = empty_tsss.filter(l_freq=l_filt, h_freq=h_filt, 
                                       picks='meg', phase=phase, 
                                       skip_by_annotation=SKIP_ANNOTATIONS,
                                       verbose=verbose)
    
    # ica
    ica = mne.preprocessing.read_ica((
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-apply-comp-ica.fif")
    )

    empty_filtered_ica = ica.apply(empty_filtered.copy())

    empty_filtered_ica.save(fname=(
        f"{megout}/{sub}"
        f"_tsss-{l_filt}-{h_filt}-{config.scan_info.session}-emptyroom"
        "-apply-raw.fif"), 
        overwrite=config.data_src.overwrite
    )

    return empty_filtered_ica


def make_src(sub, config, space, generate_bem=False, verbose=False):
    assert config.data_src.mridir is not None, \
        "MRI directory has not been initialized in pipeline_config!"

    megout, mriout = _verify_outdir(sub, config)

    if generate_bem:
        make_bem(sub, config)
    
    if 'ico' in space:
        src = mne.setup_source_space(subject=sub, spacing=space, surface='white', 
                                 subjects_dir=config.data_src.mridir,
                                 n_jobs=-1,
                                 add_dist=True, verbose=verbose)
    elif 'vol' in space:
        pos = space[3:] # e.g., vol20 yields 20 mm volume voxel grid
        bem_path = pathlib.Path(
            f"{config.data_src.mridir}/{sub}/bem/{sub}-inner_skull-bem-sol.fif"
        )
        src = mne.setup_volume_source_space(
                                        subject=sub, pos=pos, 
                                        bem=bem_path, 
                                        mindist=config.inverse.volume_mindist, 
                                        exclude=config.inverse.volume_exclude, 
                                        subjects_dir=config.data_src.mridir, 
                                        n_jobs=-1,
                                        verbose=verbose)
    else:
        raise Exception(f"source space type {space} not recognized")
    
    # convert to head coord frame to match forward
    trans_path = pathlib.Path(
        f"{config.data_src.megdir}/{sub}/{sub}-trans.fif"
    )

    assert trans_path.exists(), \
        "Trans file does not exist! Please run compute_coreg first"
    
    trans = mne.read_trans(fname = trans_path)

    src_head = mne.SourceSpaces(
        [mne.transform_surface_to(s, "head", trans, copy=True) for s in src]
    )
    
    src_head.save(fname=f"{mriout}/{sub}-{space}-src.fif", 
                    overwrite=config.data_src.overwrite)
    
    return src_head


def make_evoked(sub, config, trial, ica_apply=None, verbose=False):
    """create one trial-level evoked file and aligned annotation sidecar."""
    megout, mriout = _verify_outdir(sub, config)

    assert config.data_src.megdir is not None, (
        "MEG directory has not been initialized in pipeline_config!"
    )

    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit

    if ica_apply is None:
        raw_path = pathlib.Path(
            f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
            f"{config.scan_info.session}-ica-apply-raw.fif"
        )

        assert raw_path.exists(), (
            "ICA-applied raw file does not exist. Run apply_ica() first, "
            f"or pass ica_apply explicitly. Current path: {raw_path}"
        )

        ica_apply = mne.io.read_raw_fif(
            raw_path,
            preload=False,
            verbose=verbose,
        )

    trial_start_s = (
        config.scan_info.buffer
        + trial * config.scan_info.epoch_duration
    )
    trial_stop_s = (
        config.scan_info.buffer
        + (trial + 1) * config.scan_info.epoch_duration
    )

    if trial_start_s < 0:
        raise ValueError(
            f"Trial {trial} starts before raw time zero: {trial_start_s} s"
        )

    if trial_stop_s > ica_apply.times[-1]:
        raise ValueError(
            f"Trial {trial} extends beyond the available data. "
            f"Requested stop={trial_stop_s:.3f} s; "
            f"available stop={ica_apply.times[-1]:.3f} s."
        )

    # create the diagnostic annotation artifact before or independently of
    # cropping. its timing is set explicitly in trial-relative coordinates.
    trial_annotations = trial_relative_annotations(
        ica_apply.annotations,
        trial_start_s=trial_start_s,
        trial_stop_s=trial_stop_s,
    )

    trial_raw = ica_apply.copy().crop(
        tmin=trial_start_s,
        tmax=trial_stop_s,
    )

    print(f"Making evoked for {sub}, trial {trial}...")

    epochs = mne.make_fixed_length_epochs(
        trial_raw,
        duration=config.scan_info.epoch_duration,
        reject_by_annotation=False,
        preload=True,
        verbose=verbose,
    )

    if len(epochs) == 0:
        raise RuntimeError(
            f"No epochs were created for {sub}, trial {trial}. "
            f"Crop interval: {trial_start_s:.3f}-{trial_stop_s:.3f} s."
        )

    evoked = epochs.average()

    l_filt_narrow = config.filter_params.analysis_lower_bandlimit
    h_filt_narrow = config.filter_params.analysis_upper_bandlimit
    phase = config.filter_params.filter_phase

    evoked = evoked.filter(
        l_freq=l_filt_narrow,
        h_freq=h_filt_narrow,
        phase=phase,
        skip_by_annotation=SKIP_ANNOTATIONS,
    )

    stem = (
        f"{sub}_{config.scan_info.session}"
        f"-[trial={trial}]"
        f"-[{l_filt_narrow}-{h_filt_narrow}Hz]"
    )

    evoked_path = pathlib.Path(f"{megout}/{stem}-evoked-ave.fif")
    annotations_path = pathlib.Path(f"{megout}/{stem}-annotations.csv")

    evoked.save(
        fname=evoked_path,
        overwrite=config.data_src.overwrite,
    )

    trial_annotations.save(
        fname=annotations_path,
        overwrite=config.data_src.overwrite,
    )

    return evoked


def make_fwd(sub, config, info, space, verbose=False):
    megout, mriout = _verify_outdir(sub, config)

    # assert config.data_src.megdir is not None, \
    #     "MEG directory has not been initialized in pipeline_config!"
    
    assert config.data_src.mridir is not None, \
        "MRI directory has not been initialized in pipeline_config!"
    
    trans_path = pathlib.Path(
        f"{config.data_src.megdir}/{sub}/{sub}-trans.fif"
    )

    bem_path = pathlib.Path(
        f"{config.data_src.mridir}/{sub}/bem/{sub}-inner_skull-bem-sol.fif"
    )

    assert trans_path.exists(), \
        "Trans file does not exist! Please run compute_coreg first"
    
    assert bem_path.exists(), "Bem file does not exist! run compute_coreg first"
    src = mne.read_source_spaces(
        f"{mriout}/{sub}-{space}-src.fif"
    )

    trans = mne.read_trans(fname = trans_path)
    bem = mne.read_bem_solution(fname = bem_path)

    print("making forward")
    forward = mne.make_forward_solution(info, 
                                        trans=trans, 
                                        src=src, 
                                        bem=bem, 
                                        meg=True, 
                                        eeg=False, 
                                        ignore_ref=True, 
                                        verbose=verbose)
    print("saving forward")
    forward.save(fname=(
        f"{mriout}/{sub}_{config.scan_info.session}"
        f"-[src={space}]-solution-fwd.fif"), 
        overwrite=config.data_src.overwrite
    )

    return forward


def make_cov(sub, config, empty=None, verbose=False):
    megout, mriout = _verify_outdir(sub, config)

    if empty is None:
        l_filt = config.filter_params.wideband_lower_bandlimit
        h_filt = config.filter_params.wideband_upper_bandlimit
        assert pathlib.Path(
            f"{megout}/{sub}"
                f"_tsss-{l_filt}-{h_filt}-{config.scan_info.session}-emptyroom"
                "-apply-raw.fif").exists(), \
            ("Empty room does not exist. Please either pass a valid instance "
            "of tsss filtered empty room or run filter_empty()")
        empty=mne.io.read_raw_fif(f"{megout}/{sub}"
                f"_tsss-{l_filt}-{h_filt}-{config.scan_info.session}-emptyroom"
                "-apply-raw.fif", preload=True)
    
    # drop transients at start and end
    empty = empty.crop(tmin=config.scan_info.buffer, tmax=empty.times[-1] - \
                       config.scan_info.buffer)

    l_filt_narrow = config.filter_params.analysis_lower_bandlimit
    h_filt_narrow = config.filter_params.analysis_upper_bandlimit
    phase = config.filter_params.filter_phase
    empty.filter(l_freq=l_filt_narrow, h_freq=h_filt_narrow, 
                 skip_by_annotation=SKIP_ANNOTATIONS,
                 phase=phase)

    cov = mne.compute_raw_covariance(empty, picks='meg', 
                                     method='auto', rank='info')
    
    mne.write_cov(
        fname=f"{megout}/{sub}-[{l_filt_narrow}-{h_filt_narrow}Hz]-cov.fif", 
        cov=cov, 
        overwrite=config.data_src.overwrite
    )
    
    return cov


def view_ica(sub, config):
    megout, mriout = _verify_outdir(sub, config)

    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit

    tsss_causal = mne.io.read_raw_fif((
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-raw.fif"))
    
    ica = mne.preprocessing.read_ica((
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-ica.fif"))
    
    ica.plot_components(list(range(30)))
    ica.plot_sources(tsss_causal, block=True)


def _verify_outdir(sub, config):
    if config.data_src.outdir is None: 
        return config.data_src.megdir / f"{sub}", \
            config.data_src.mridir / f"{sub}" / 'bem'
    
    megout = config.data_src.outdir / 'meg_dir' / f"{sub}"
    bemout = config.data_src.outdir / 'mri_dir' / f"{sub}" / 'bem'
    megout.mkdir(parents=True, exist_ok=True)
    bemout.mkdir(parents=True, exist_ok=True)
    return megout, bemout
