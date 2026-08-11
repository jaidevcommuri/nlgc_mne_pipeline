import mne
import pathlib
import numpy as np
from dataclasses import dataclass
from config import pipeline_config
from mne_icalabel import label_components


def fit_ica(sub, config, components=None, verbose=False):
    assert config.megdir != None, "MEG directory has not been initialized in pipeline_config!"
    assert config.session != 'UNK', "Please initialize session in the configuration!"
    #assert config.meg_sys != 'KIT' and config.meg_sys != 'MEGIN', "Please enter a valid MEG sensor system (Current valid systems: KIT, MEGIN)"

    if (config.meg_sys=='KIT'):
        st_only=True
    elif (config.meg_sys=='MEGIN'):
        st_only=False
    else:
        raise RuntimeError("Invalid MEG System in config!")

    raw_path = pathlib.Path(f"{config.megdir}/{sub}/{sub}_resting_{config.session}-raw.fif")
    assert raw_path.exists() == True, "Raw MEG file does not exist!"
    if (components==None):
        components=config.components

    raw = mne.io.read_raw_fif(raw_path, preload=True)
    raw.del_proj() # Incase we are working with MEGIN data, we need to get rid of projection vectors before ICA
    raw = raw.pick(picks='meg', exclude='bads')
    noisy_ch, flat_ch = mne.preprocessing.find_bad_channels_maxwell(raw, coord_frame='head', ignore_ref=True, verbose=verbose)
    raw.info['bads'] += noisy_ch + flat_ch
    raw_tsss = mne.preprocessing.maxwell_filter(raw, st_duration=10, ignore_ref=True, st_correlation=0.9, st_only=st_only, bad_condition='warning', coord_frame='head', verbose=verbose)
    tsss_causal = raw_tsss.filter(l_freq=1, h_freq=45, picks='meg', phase='minimum', verbose=verbose)
    tsss_causal.save(fname=f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting-{config.session}-raw.fif", overwrite=config.overwrite)
    ica = mne.preprocessing.ICA(n_components=config.components)
    ica.fit(tsss_causal)
    ica.save(f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting-{config.session}-ica.fif", overwrite=config.overwrite)
    return tsss_causal, ica

def apply_ica(sub, config, tsss_causal=None, ica=None, verbose=False):
    assert config.megdir != None, "MEG directory has not been initialized in pipeline_config!"
    assert config.session != 'UNK', "Please initialize session in the configuration!"
    tsss_path = pathlib.Path(f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting-{config.session}-raw.fif")
    ica_path = pathlib.Path(f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting-{config.session}-ica.fif")
    if (tsss_causal==None):
        assert tsss_path.exists() == True, "Must fit ICA before applying. Run fit_ica() before this function"
        tsss_causal = mne.io.read_raw_fif(tsss_path, preload=True)
    if (ica==None):
        assert ica_path.exists() == True, "Must fit ICA before applying. Run fit_ica() before this function"
        ica = mne.preprocessing.read_ica(ica_path)


    if (config.ICA_mode=='manual'):
        return _manual_ica(sub, config, tsss_causal, ica)
    elif(config.ICA_mode=='auto'):
        return _auto_ica(sub, config, tsss_causal, ica)
    else:
        raise RuntimeError("config.ICA_mode set up incorrectly! Please set it to either manual or auto")

def _manual_ica(sub, config, tsss_causal, ica):
    ica.plot_components(list(range(20)))
    ica.plot_sources(tsss_causal, block=True)
    ica.save(f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting-{config.session}-apply-comp-ica.fif", overwrite=config.overwrite)
    ica_apply = ica.apply(tsss_causal.copy(), exclude=ica.exclude)
    ica_apply.save(f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting-{config.session}-ica-apply-raw.fif", overwrite=config.overwrite)
    return ica_apply

def _auto_ica(sub, config, tsss_causal, ica):
    labels = label_components(tsss_causal, ica, method='megnet')
    print(labels)
    for idx, label in enumerate(labels['labels']):
        if label in ['eye blink', 'heart beat']:
            ica.exclude.append(idx)

    ica_apply = ica.apply(tsss_causal.copy(), exclude=ica.exclude)
    ica_apply.save(f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting-{config.session}-ica-apply-raw.fif", overwrite=config.overwrite)
    ica.save(f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting-{config.session}-apply-comp-ica.fif", overwrite=config.overwrite)
    return ica_apply


def filter_empty(sub, config, verbose=False):
    print("Filtering emptyroom...")
    assert config.megdir != None, "MEG directory has not been initialized in pipeline_config!"
    empty_raw_path = pathlib.Path(f"{config.megdir}/{sub}/{sub}_emptyroom-raw.fif")
    assert empty_raw_path.exists() == True, "Raw emptyroom does not exist!"

    if (config.meg_sys=='KIT'):
        st_only=True
    elif (config.meg_sys=='MEGIN'):
        assert pathlib.Path(f"{config.megdir}/{sub}/{sub}_resting_{config.session}-raw.fif").exists(), "MEGIN systems require info from the raw MEG file to be properly oriented! Raw file not found!"
        st_only=False
    else:
        raise RuntimeError("Invalid MEG System in config!")

    raw_empty = mne.io.read_raw_fif(empty_raw_path, preload=True)
    raw_empty.del_proj() # Necessary for some MEG Systems to be compatible
    raw_empty = raw_empty.pick(picks='meg', exclude='bads')
    noisy_ch, flat_ch = mne.preprocessing.find_bad_channels_maxwell(raw_empty, coord_frame='meg', ignore_ref=True, verbose=verbose)
    raw_empty.info['bads'] += noisy_ch + flat_ch
    if config.meg_sys=='MEGIN':
        raw = mne.io.load_raw_fiif(f"{config.megdir}/{sub}/{sub}_resting_{config.session}-raw.fif", preload=False)
        raw_empty.info['dev_head_t'] = raw.info['dev_head_t'] # Causes Null Space malalignment if not present for covariance
    empty_tsss = mne.preprocessing.maxwell_filter(raw_empty, st_duration=10, ignore_ref=True, st_correlation=0.9, st_only=st_only, bad_condition='warning', coord_frame='head', verbose=verbose)
    empty_causal = empty_tsss.filter(l_freq=1, h_freq=45, picks='meg', phase='minimum', verbose=verbose)
    empty_causal.save(fname=f"{config.megdir}/{sub}/{sub}_tsss-emptyroom.fif", overwrite=config.overwrite)
    return empty_causal

# def compute_coreg(): 
    # Not yet implemented

def make_src(sub, config, space, verbose=False):
    assert config.mridir != None, "MRI directory has not been initialized in pipeline_config!"
    src = mne.setup_source_space(subject=sub, spacing=space, surface='white', subjects_dir=config.mridir, add_dist=True, verbose=verbose)
    src.save(fname=f"{config.mridir}/{sub}/bem/{sub}_{space}-src.fif", overwrite=config.overwrite)
    return src

def make_evoked(sub, config, trial, ica_apply=None, verbose=False):
    if (ica_apply==None):
        assert pathlib.Path(f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting_{config.session}-ica-apply-raw.fif").exists(), "ica_apply does not exist in memory. Please either pass a valid instance of applied ica to raw data, or run apply_ica()"
        ica_apply = mne.io.read_raw_fif(f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting-{config.session}-ica-apply-raw.fif")
    assert config.megdir != None, "MEG directory has not been initialized in pipeline_config!"
    print("Making epochs...")
    ica_apply = ica_apply.crop(tmin=config.buffer + trial * config.duration, tmax=config.buffer + (trial + 1) * config.duration)
    epochs = mne.make_fixed_length_epochs(ica_apply, duration=config.epoch_duration, reject_by_annotation=False, preload=True, verbose=verbose)
    print(epochs)
    print(type(epochs))
    # decimation may not be necessary
    # epochs.decimate(decim=10)
    evoked = epochs.average()
    # print(evoked_data.nave)
    evoked = evoked.filter(l_freq=13, h_freq=25, phase='minimum')
    evoked.save(fname=f"{config.megdir}/{sub}/{sub}_{config.session}-[trial:{trial}]-evoked-ave.fif", overwrite=config.overwrite)
    return evoked

# Space when none defaults to ico4
def make_fwd(sub, config, evoked, trial, space=None, verbose=False):
    assert config.megdir != None, "MEG directory has not been initialized in pipeline_config!"
    assert config.mridir != None, "MRI directory has not been initialized in pipeline_config!"
    trans_path = pathlib.Path(f"{config.megdir}/{sub}/{sub}-trans.fif")
    bem_path = pathlib.Path(f"{config.mridir}/{sub}/bem/{sub}-inner_skull-bem-sol.fif")
    assert trans_path.exists(), "Trans file does not exist! Please run compute_coreg first"
    assert bem_path.exists(), "Bem files does not exist! Please run compute_coreg first"

    trans = mne.read_trans(fname=trans_path)
    bem = mne.read_bem_solution(fname=bem_path)
    forward = mne.make_forward_solution(evoked.info, trans=trans, src=space, bem=bem, meg=True, eeg=False, ignore_ref=True, verbose=verbose)
    forward.save(fname=f"{config.mridir}/{sub}/bem/{sub}_{config.session}-[trial:{trial}]-solution-fwd.fif", overwrite=config.overwrite)
    return forward

def make_cov(sub, config, empty=None, verbose=False):
    assert config.megdir != None, "MEG directory has not been initialized in pipeline_config!"
    if (empty==None):
        assert pathlib.Path(f"{config.megdir}/{sub}/{sub}_tsss-emptyroom.fif").exists(), "Empty room does not exist. Please either pass a valid instance of tsss filtered empty room or run filter_empty()"
    empty = empty.crop(tmin=config.buffer, tmax=empty.times[-1] - config.buffer)
    cov = mne.compute_raw_covariance(empty, picks='meg', method='auto', rank='info')
    mne.write_cov(fname=f"{config.megdir}/{sub}/{sub}_cov.fif", cov=cov, overwrite=config.overwrite)
    return cov

def view_ica(sub, config):
    tsss_causal = mne.io.read_raw_fif(f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting-{config.session}-raw.fif")
    ica = mne.preprocessing.read_ica(f"{config.megdir}/{sub}/{sub}_tsss-1-45-causal-resting-{config.session}-ica.fif")
    ica.plot_components(list(range(30)))
    ica.plot_sources(tsss_causal, block=True)
