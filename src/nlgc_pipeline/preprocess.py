import mne
import pathlib
import numpy as np
from mne_icalabel import label_components
from mne.coreg import Coregistration
from nlgc_pipeline.utils.surfaces import (make_scalp_surfaces, 
                                          make_bem,
                                          _remove_implausible_dig_pts)
                                          

def compute_coreg(sub, config, raw, verbose=False): 
    trans_path = pathlib.Path(
        f"{config.data_src.megdir}/{sub}/{sub}-trans.fif"
    )

    # coreg already done (or manually redone)
    if trans_path.exists():
        return raw
    
    # need scalp surfaces for coreg
    make_scalp_surfaces(sub, subjects_dir=config.data_src.mridir)
    
    # we generally don't have the exact fiducial points and want to avoid
    # pulling up the gui to set them manually. we use the fsaverage fiducial
    # locations as the first pass and then fit ICP to the hsp to refine the
    # overall fit.
    coreg = Coregistration(
        raw.info,
        sub,
        subjects_dir=config.data_src.mridir,
        fiducials="estimated",
    )

    coreg.fit_fiducials(verbose=verbose)
    coreg.fit_icp(
        n_iterations=10,
        nasion_weight=2.0, # decrease nasion weight as fiducials are estimated
    )

    # !!! important for tsss, which fits a sphere centered at head origin
    # defined by hsp. identify and drop implausible hsp !!!
    raw, coreg = _remove_implausible_dig_pts(raw, coreg)

    # nasion weight higher since we are close to optimum
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

    # fit info
    dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
    print(
        f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm"
        f" / {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
    )

    mne.write_trans(trans_path, coreg.trans)

    return raw


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

    # auto coregister AND drop implausible dig points, which can be important
    # for maxwell fitering in next step
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
    raw = raw.pick(picks='meg', exclude='bads')
    noisy_ch, flat_ch = mne.preprocessing.find_bad_channels_maxwell(raw, 
                                                            coord_frame='head', 
                                                            ignore_ref=True, 
                                                            verbose=verbose)
    raw.info['bads'] += noisy_ch + flat_ch

    # maxwell filter
    if config.verbose:
        print("maxwell filtering")
    raw_tsss = mne.preprocessing.maxwell_filter(raw, st_duration=10, 
                                                ignore_ref=True, 
                                                st_correlation=0.9, 
                                                st_only=st_only, 
                                                bad_condition='warning', 
                                                coord_frame='head', 
                                                verbose=verbose)
    
    # bandpass filter
    if config.verbose:
        print("wide band filtering")
    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit
    phase = config.filter_params.filter_phase
    tsss_causal = raw_tsss.filter(l_freq=l_filt, h_freq=h_filt, picks='meg', 
                                  phase=phase, verbose=verbose)
   
    tsss_causal.save(fname=(
        f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
        f"{config.scan_info.session}-raw.fif"), 
        overwrite=config.data_src.overwrite
    )

    # ica fit
    if config.verbose:
        print("fit ica")
    ica = mne.preprocessing.ICA(
        n_components = config.filter_params.ICA_components,
        method = config.filter_params.ICA_method,
    )
    
    ica.fit(tsss_causal)

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
                                                  verbose=verbose)
    
    # bandpass filter
    phase = config.filter_params.filter_phase
    empty_filtered = empty_tsss.filter(l_freq=l_filt, h_freq=h_filt, 
                                       picks='meg', phase=phase, 
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


def make_src(sub, config, space, verbose=False):
    assert config.data_src.mridir is not None, \
        "MRI directory has not been initialized in pipeline_config!"

    megout, mriout = _verify_outdir(sub, config)

    # need surfaces for forward later and possibly for vol source space mask now
    make_bem(sub, config)
    
    if 'ico' in space:
        src = mne.setup_source_space(subject=sub, spacing=space, surface='white', 
                                 subjects_dir=config.data_src.mridir,
                                 add_dist=True, verbose=verbose)
    elif 'vol' in space:
        pos = space[3:] # e.g., vol20 yields 20 mm volume voxel grid
        bem_path = pathlib.Path(
            f"{config.data_src.mridir}/{sub}/bem/{sub}-inner_skull-bem-sol.fif"
        )
        src = mne.setup_volume_source_space(subject=sub, pos=pos, 
                                            bem=bem_path, 
                                            mindist=5.0, exclude=0.0, 
                                            subjects_dir=config.data_src.mridir, 
                                            verbose=verbose)
    else:
        raise Exception(f"source space type {space} not recognized")
    
    src.save(fname=f"{mriout}/{sub}_{space}-src.fif", 
             overwrite=config.data_src.overwrite)
    return src


def make_evoked(sub, config, trial, ica_apply=None, verbose=False):

    megout, mriout = _verify_outdir(sub, config)

    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit

    assert config.data_src.megdir is not None, \
        "MEG directory has not been initialized in pipeline_config!"

    if ica_apply is None:
        assert pathlib.Path((
            f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
            f"{config.scan_info.session}-apply-comp-ica.fif")).exists(), \
        ("ica_apply does not exist in memory. Please either pass a valid "
         "instance of applied ica to raw data, or run apply_ica()")
        
        ica_apply = mne.io.read_raw_fif((
            f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-"
            f"{config.scan_info.session}-ica-apply-raw.fif"))
    
    print("Making epochs...")
    ica_apply = ica_apply.crop(
        tmin=config.scan_info.buffer + trial * \
            config.scan_info.epoch_duration, 
        tmax=config.scan_info.buffer + (trial + 1) * \
            config.scan_info.epoch_duration
    )

    epochs = mne.make_fixed_length_epochs(ica_apply, 
                                    duration=config.scan_info.epoch_duration, 
                                    reject_by_annotation=False, 
                                    preload=True, 
                                    verbose=verbose)
    print(epochs)
    print(type(epochs))

    evoked = epochs.average()

    l_filt_narrow = config.filter_params.analysis_lower_bandlimit
    h_filt_narrow = config.filter_params.analysis_upper_bandlimit
    phase = config.filter_params.filter_phase
    evoked = evoked.filter(l_freq=l_filt_narrow, h_freq=h_filt_narrow, 
                           phase=phase)

    evoked.save(fname=(
            f"{megout}/{sub}_{config.scan_info.session}"
            f"-[trial={trial}]-evoked-ave.fif"), 
            overwrite=config.data_src.overwrite
    )
    return evoked


def make_fwd(sub, config, evoked, trial, space, verbose=False):
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
    space = mne.read_source_spaces(
        f"{mriout}/{sub}_{space}-src.fif"
    )

    trans = mne.read_trans(fname = trans_path)
    bem = mne.read_bem_solution(fname = bem_path)

    print("making forward")
    forward = mne.make_forward_solution(evoked.info, 
                                        trans=trans, 
                                        src=space, 
                                        bem=bem, 
                                        meg=True, 
                                        eeg=False, 
                                        ignore_ref=True, 
                                        verbose=verbose)
    print("saving forward")
    forward.save(fname=(
        f"{mriout}/{sub}_{config.scan_info.session}"
        f"-[trial={trial}]-[src={space}]-solution-fwd.fif"), 
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
                           phase=phase)

    cov = mne.compute_raw_covariance(empty, picks='meg', 
                                     method='auto', rank='info')
    
    mne.write_cov(fname=f"{megout}/{sub}_cov.fif", 
                  cov=cov, overwrite=config.data_src.overwrite)
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
