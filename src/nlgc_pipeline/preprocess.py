import mne
import pathlib
import numpy as np
from dataclasses import dataclass, asdict
from mne_icalabel import label_components
import hashlib
import json
import dacite
from nlgc_pipeline import config, checksum



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
    
    assert raw_path.exists(), f"Raw MEG file does not exist! Current file path: {raw_path}"

    for trial in config.scan_info.trials:
        config.metadata.save_checksum(pipeline=config.metadata.resting_pipe, data_id='Raw', \
            trial=trial, session=config.scan_info.session, dst=raw_path)

    # config.metadata.alias_data(src_pipeline=config.metadata.resting_pipe,\
    #     src_id='Raw', target_pipeline=config.metadata.forward, target_id='Raw')

    megout, mriout = _verify_outdir(sub, config)
    
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    raw = raw.resample(config.filter_params.sfreq)

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

    tsss_path = f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-" \
                f"{config.scan_info.session}-raw.fif"
   
    tsss_causal.save(fname=tsss_path, 
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

    ica_path = f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-" \
            f"{config.scan_info.session}-ica.fif" 

    ica.save(ica_path, overwrite=config.data_src.overwrite)

    for trial in config.scan_info.trials:
        config.metadata.save_checksum(pipeline=config.metadata.resting_pipe, data_id='Tsss', \
            trial=trial, session=config.scan_info.session, dst=tsss_path)
        config.metadata.save_checksum(pipeline=config.metadata.resting_pipe, data_id='ICA', \
            trial=trial, session=config.scan_info.session, dst=ica_path)

    # _save_checksum(config, raw_path)
    # _save_checksum(config, tsss_path)
    # _save_checksum(config, ica_path)

    return tsss_causal, ica

def apply_ica(sub, config, tsss_causal=None, ica=None, verbose=False):
    if config.verbose:
        print(f"apply ica with mode={config.filter_params.ICA_mode}")
    assert config.data_src.megdir is not None, \
        "MEG directory has not been initialized in pipeline_config!"
    assert config.scan_info.session != 'UNK', \
        "Please initialize session in the configuration!"

    config.metadata.verify_data(pipeline=config.metadata.resting_pipe, \
        data_id='ICA Comp', session=config.scan_info.session)


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
    else:
        raise RuntimeError(("config.ICA_mode set up incorrectly! "
                           "Please set it to either manual or auto"))

def _manual_ica(sub, config, tsss_causal, ica, megout):
    ica.plot_components(list(range(20)))
    ica.plot_sources(tsss_causal, block=True)

    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit

    ica_path = f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-" \
            f"{config.scan_info.session}-apply-comp-ica.fif"

    ica.save(ica_path, overwrite=config.data_src.overwrite)
    ica_apply = ica.apply(tsss_causal.copy(), exclude=ica.exclude)
    apply_path = f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-" \
            f"{config.scan_info.session}-ica-apply-raw.fif"
    ica_apply.save(apply_path, overwrite=config.data_src.overwrite)

    for trial in config.scan_info.trials:
        config.metadata.save_checksum(pipeline=config.metadata.resting_pipe, data_id='ICA Comp', \
            trial=trial, session=config.scan_info.session, dst=ica_path)
        config.metadata.save_checksum(pipeline=config.metadata.resting_pipe, data_id='Apply', \
            trial=trial, session=config.scan_info.session, dst=apply_path)
    # config.metadata.alias_data(src_pipeline=config.metadata.resting_pipe, \
    #     src_id='ICA Comp', target_pipeline=config.metadata.empty_pipe, \
    #     target_id='ICA Comp')
    
    return ica_apply

def _auto_ica(sub, config, tsss_causal, ica, megout):
    l_filt = config.filter_params.wideband_lower_bandlimit
    h_filt = config.filter_params.wideband_upper_bandlimit

    labels = label_components(tsss_causal, ica, method='megnet')
    print(labels)
    for idx, label in enumerate(labels['labels']):
        if label in ['eye blink', 'heart beat']:
            ica.exclude.append(idx)

    ica_path = f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-" \
            f"{config.scan_info.session}-apply-comp-ica.fif"

    apply_path = f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-" \
            f"{config.scan_info.session}-ica-apply-raw.fif"

    ica_apply = ica.apply(tsss_causal.copy(), exclude=ica.exclude)
    ica_apply.save(apply_path, overwrite=config.data_src.overwrite)
    
    ica.save(ica_path, overwrite=config.data_src.overwrite)
    

    for trial in config.scan_info.trials:
        config.metadata.save_checksum(pipeline=config.metadata.resting_pipe, data_id='ICA Comp', \
            trial=trial, session=config.scan_info.session, dst=ica_path)
        config.metadata.save_checksum(pipeline=config.metadata.resting_pipe, data_id='Apply', \
            trial=trial, session=config.scan_info.session, dst=apply_path)
    # config.metadata.alias_data(src_pipeline=config.metadata.resting_pipe, \
    #     src_id='ICA Comp', target_pipeline=config.metadata.empty_pipe, \
    #     target_id='ICA Comp')
    
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

    config.metadata.verify_data(pipeline=config.metadata.empty_pipe, \
        data_id='Raw', session=config.scan_info.session)

    for trial in config.scan_info.trials:
            config.metadata.save_checksum(pipeline=config.metadata.empty_pipe, data_id='Raw', \
                trial=trial, session=config.scan_info.session, dst=empty_raw_path)
    
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

    tsss_path = f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-" \
                f"{config.scan_info.session}-emptyroom-tsss-raw.fif"

    
    
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

    empty_path = f"{megout}/{sub}_tsss-{l_filt}-{h_filt}-" \
            f"{config.scan_info.session}-emptyroom-apply-raw.fif"

    empty_tsss.save(tsss_path, overwrite=config.data_src.overwrite)
    empty_filtered_ica.save(empty_path, overwrite=config.data_src.overwrite)

    for trial in config.scan_info.trials:
        config.metadata.save_checksum(pipeline=config.metadata.empty_pipe, data_id='Tsss', \
            trial=trial, session=config.scan_info.session, dst=tsss_path)
        config.metadata.save_checksum(pipeline=config.metadata.empty_pipe, data_id='Apply', \
            trial=trial, session=config.scan_info.session, dst=empty_path)
    
    return empty_filtered_ica

# def compute_coreg(): 
    # Not yet implemented

def make_src(sub, config, space, verbose=False):
    assert config.data_src.mridir is not None, \
        "MRI directory has not been initialized in pipeline_config!"

    megout, mriout = _verify_outdir(sub, config)
    
    src = mne.setup_source_space(subject=sub, spacing=space, surface='white', 
                                 subjects_dir=config.data_src.mridir,
                                 add_dist=True, verbose=verbose)

    src_path = f"{mriout}/{sub}_{space}-src.fif"
    
    src.save(fname=src_path, overwrite=config.data_src.overwrite)

    for trial in config.scan_info.trials:
        if space not in config.metadata.forward: continue
        config.metadata.save_checksum(pipeline=config.metadata.forward, data_id=space, \
            trial=trial, session=config.scan_info.session, dst=src_path)
    
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

    evoked_path = f"{megout}/{sub}_{config.scan_info.session}" \
            f"-[trial={trial}]-evoked-ave.fif"

    evoked.save(fname=evoked_path, overwrite=config.data_src.overwrite)

    config.metadata.save_checksum(pipeline=config.metadata.resting_pipe, data_id='Evoked', \
        trial=trial, session=config.scan_info.session, dst=evoked_path)
    
    return evoked

# Space when none defaults to ico4
def make_fwd(sub, config, trial, space=None, verbose=False):
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
    if space is None:
        space = mne.read_source_spaces(
            f"{mriout}/{sub}_ico4-src.fif"
        )

    config.metadata.save_checksum(pipeline=config.metadata.forward, data_id='Trans', \
        trial=trial, session=config.scan_info.session, dst=trans_path)
    config.metadata.save_checksum(pipeline=config.metadata.forward, data_id='Bem', \
        trial=trial, session=config.scan_info.session, dst=bem_path)

    config.metadata.verify_data(pipeline=config.metadata.forward, data_id='Forward', trial=trial)

    raw = mne.io.read_raw_fif(pathlib.Path(config.metadata.find_node(pipeline=config.metadata.forward, data_id='Raw',
                                trial=trial, session=config.scan_info.session).file))

    trans = mne.read_trans(fname = trans_path)
    bem = mne.read_bem_solution(fname = bem_path)

    print("making forward")
    forward = mne.make_forward_solution(raw.info, 
                                        trans=trans, 
                                        src=space, 
                                        bem=bem, 
                                        meg=True, 
                                        eeg=False, 
                                        ignore_ref=True, 
                                        verbose=verbose)
    print("saving forward")
    fwd_path = f"{mriout}/{sub}_{config.scan_info.session}" \
            f"-[trial={trial}]-solution-fwd.fif"
    forward.save(fname=fwd_path, overwrite=config.data_src.overwrite)

    config.metadata.save_checksum(pipeline=config.metadata.forward, data_id='Forward', \
        trial=trial, session=config.scan_info.session, dst=fwd_path)
    
    return forward

def make_cov(sub, config, empty=None, verbose=False):
    megout, mriout = _verify_outdir(sub, config)
    # assert config.data_src.megdir is not None, \
    #     "MEG directory has not been initialized in pipeline_config!"

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

    config.metadata.verify_data(pipeline=config.metadata.empty_pipe, data_id='Cov')
    
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

    cov_path = f"{megout}/{sub}_cov.fif"
    
    mne.write_cov(fname=cov_path, 
                  cov=cov, overwrite=config.data_src.overwrite)

    for trial in config.scan_info.trials:
        config.metadata.save_checksum(pipeline=config.metadata.empty_pipe, data_id='Cov', \
            trial=trial, session=config.scan_info.session, dst=cov_path)
    
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
    if config.data_src.outdir is None: return f"{config.data_src.megdir}/{sub}", f"{config.data_src.mridir}/{sub}/bem"
    megout = f"{config.data_src.outdir}/meg/{sub}"
    bemout = f"{config.data_src.outdir}/meg/{sub}/bem"
    pathlib.Path(megout).mkdir(parents=True, exist_ok=True)
    pathlib.Path(bemout).mkdir(parents=True, exist_ok=True)
    return megout, bemout

def load_config(src):

    assert pathlib.Path(src).exists(), f"Error: cannot find file {src}"

    with open(f"{src}", "r") as f:
        data = json.load(f)
        # print(data)
    metadata = data['metadata']
    pipeline_list = data['metadata'].keys()
    print(pipeline_list)
    # config_ret = dacite.from_dict(data_class=config.PipelineConfig, data=data)
    # return config_ret
    for pipeline in pipeline_list:
        if type(metadata[pipeline]) != dict: continue
        for key in metadata[pipeline]:
            if key == 'Mark': continue
            if type(metadata[pipeline][key]) is str: continue
            metadata[pipeline][key] = [checksum.PipelineNode(**d) \
                for d in metadata[pipeline][key]]
    data['metadata'] = metadata
    return dacite.from_dict(data_class=config.PipelineConfig, data=data)
    # config = dacite.from_dict(data_class=config.PipelineConfig, data=data)
    # config.metadata

    
    
