import mne
import pathlib
import eelfarm
import eelbrain as eel
import nlgc
import preprocess
from config import pipeline_config

def preprocess_pipeline(sub, config, verbose=False):
    tsss_causal, ica = preprocess.fit_ica(sub, config, verbose)
    ica_apply = preprocess.apply_ica(sub, config, tsss_causal=tsss_causal, ica=ica, verbose=verbose)
    print(type(ica_apply))
    empty = preprocess.filter_empty(sub, config, verbose=verbose)

    for space in config.source_spaces:
        preprocess.make_src(sub, config, space=space, verbose=verbose)

    for trial in config.trials:
        evoked = preprocess.make_evoked(sub, config, ica_apply=ica_apply, trial=trial, verbose=verbose)
        preprocess.make_fwd(sub, config, evoked=evoked, trial=trial, verbose=verbose)

    preprocess.make_cov(sub, config, empty=empty, verbose=verbose)

    return 

def start_nlgc(sub, config, trial, eelfarm=False):
    src = mne.read_source_spaces(f"{config.mridir}/{sub}/bem/{sub}_ico1-src.fif")
    evoked = preprocess.make_evoked(sub, config, trial, verbose=False) # Need to figure out how to properly load evoked data from memory
    forward = mne.read_forward_solution(f"{config.mridir}/{sub}/bem/{sub}_{config.session}-[trial={trial}]-solution-fwd.fif")
    fwd = mne.convert_forward_solution(forward, force_fixed=True)
    cov = mne.read_cov(f"{config.megdir}/{sub}/{sub}_cov.fif")

    lam=[0.1]
    result_dir=pathlib.Path(f"/Volumes/PortableSSD/Alzheimers/results")
    dst = f"{result_dir}/beta25hz/4eig2ord/[{sub}]-[visit=]-[session={config.session}]-[trial={trial}]-[beta]-[fullmodel].p"

    if eelfarm:
        # TODO: server not initialized via server = eelfarm.start_server
        print("nlgc start")
        server.put(dst, nlgc.nlgc_map, sub, evoked, fwd, cov, src, use_es=False,
            order=2, n_eigenmodes=4, patch_idx=list(range(84)),
            lambda_range=lam, max_iter=500, max_cyclic_iter=3,
            tol=1e-5, sparsity_factor=0.0, var_thr=1.0, cv=5, 
            n_orients=1, parallel_mode='multiprocess', n_workers=12,
            verbose=True)
    else:
        obj = nlgc.nlgc_map(sub, evoked, fwd, cov, src, use_es=False,
            order=2, n_eigenmodes=4, patch_idx=list(range(84)),
            lambda_range=lam, max_iter=500, max_cyclic_iter=3,
            tol=1e-5, sparsity_factor=0.0, var_thr=1.0, cv=5, 
            n_orients=1, parallel_mode='multiprocess', n_workers=12,
            verbose=True)

        eel.save.pickle(obj, dst)


config = pipeline_config(
        megdir=pathlib.Path(f"/Volumes/PortableSSD/Alzheimers/meg_dir"),
        mridir=pathlib.Path(f"/Volumes/PortableSSD/Alzheimers/mri_dir"),

        # evoked cannot be properly saved and thus must be made directly before use
        meg_sys='MEGIN',
        session='EC1', 
        overwrite=True,
        make_resting=True,
        make_empty=True,
        make_ICA=True,
        ICA_mode='auto',
        make_covariance=True,
        source_spaces=['ico1', 'ico4']
    )

sub_count = 0

for subfp in config.megdir.glob(f"A*"):
    if (sub_count >= 1):
        break
    sub = subfp.name
    # preprocess_pipeline(sub, config, verbose=False)
    for trial in config.trials:
        start_nlgc(sub, config, trial, eelfarm=False)
        break
    sub_count += 1


print("finish")
