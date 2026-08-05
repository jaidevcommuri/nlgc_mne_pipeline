import mne
import pathlib
import eelfarm
import nlgc
import numpy
import eelbrain as eel

root = pathlib.Path(f"/Volumes/PortableSSD/Alzheimers")
megdir = root / 'meg_dir'
mridir = root / 'mri_dir'
result_dir = root / 'results'


# meg data was derived from KIT system
def preprocess(sub, session, start=0, duration=60, return_vars=True, resting=False, empty=False, ico1=False, ico4=False, fwd=False, covariance=False):
    
    resting_path = pathlib.Path(f"{megdir}/{sub}/{sub}_tsss-1-45-causal-resting_{session}-ica-apply.fif")
    empty_path = pathlib.Path(f"{megdir}/{sub}/{sub}_tsss-emptyroom.fif")
    ico1_path = pathlib.Path(f"{mridir}/{sub}/bem/{sub}_ico-1-src.fif")
    ico4_path = pathlib.Path(f"{mridir}/{sub}/bem/{sub}_ico-4-src.fif")
    evoked_path = pathlib.Path(f"{megdir}/{sub}/{sub}_{session}_evoked-ave.fif")
    forward_path = pathlib.Path(f"{mridir}/{sub}/bem/{sub}_{session}-solution-fwd.fif")
    trans_path = pathlib.Path(f"{megdir}/{sub}/{sub}-trans.fif")
    bem_path = pathlib.Path(f"{mridir}/{sub}/bem/{sub}-inner_skull-bem-sol.fif")
    cov_path = pathlib.Path(f"{megdir}/{sub}/{sub}_cov.fif")

    assert trans_path.exists()
    assert bem_path.exists()

    if (resting==True or resting_path.exists()==False):
        raw = mne.io.read_raw_fif(f"{megdir}/{sub}/{sub}_resting_{session}-raw.fif", preload=True)
        raw = raw.pick(picks='meg', exclude='bads')
        noisy_ch, flat_ch = mne.preprocessing.find_bad_channels_maxwell(raw, coord_frame='head', ignore_ref=True, verbose=True)
        raw.info['bads'] += noisy_ch + flat_ch
        raw_tsss = mne.preprocessing.maxwell_filter(raw, st_duration=10, ignore_ref=True, st_correlation=0.9, st_only=True, bad_condition='warning', coord_frame='head', verbose=True)
        tsss_causal = raw_tsss.filter(l_freq=1, h_freq=45, picks='meg', phase='minimum', verbose=True)
        tsss_causal.save(fname=resting_path, overwrite=True)

        ica = mne.preprocessing.ICA(n_components=30)
        ica.fit(tsss_causal)
        ica.plot_components(list(range(20)))
        ica.plot_sources(tsss_causal, block=True)
        ica.save(f"{megdir}/{sub}/{sub}_tsss-1-45-causal-resting_{session}-ica-apply-comp.fif", overwrite=True)
        ica_apply = ica.apply(tsss_causal.copy(), exclude=ica.exclude)
        ica_apply.save(f"{megdir}/{sub}/{sub}_tsss-1-45-causal-resting_{session}-ica-apply.fif", overwrite=True)
    else:
        ica_apply = mne.io.read_raw_fif(fname=resting_path, preload=True)

    if (empty==True or empty_path.exists()==False):
        raw_empty = mne.io.read_raw_fif(f"{megdir}/{sub}/{sub}_emptyroom-raw.fif", preload=True)
        raw_empty = raw_empty.pick(picks='meg', exclude='bads')
        noisy_ch, flat_ch = mne.preprocessing.find_bad_channels_maxwell(raw_empty, coord_frame='meg', ignore_ref=True, verbose=True)
        raw.info['bads'] += noisy_ch + flat_ch
        empty_tsss = mne.preprocessing.maxwell_filter(raw_empty, st_duration=10, ignore_ref=True, st_correlation=0.9, st_only=True, bad_condition='warning', coord_frame='meg', verbose=True)
        empty_causal = empty_tsss.filter(l_freq=1, h_freq=45, picks='meg', phase='minimum', verbose=True)
        empty_causal.save(fname=empty_path, overwrite=True)
    else:
        empty_causal = mne.io.read_raw_fif(empty_path, preload=True)
    
    if (ico1==True or ico1_path.exists()==False):
        # white and pial both possible surfaces. pial may be better
        # add dist may or may not affect final results. More testing is needed
        src1 = mne.setup_source_space(subject=sub, spacing='ico1', surface='white', subjects_dir=mridir, add_dist=True, verbose=True)
        src1.save(fname=ico1_path, overwrite=True)
    else:
        src1 = mne.read_source_spaces(fname=ico1_path)

    if (ico4==True or ico4_path.exists()==False):
        # white and pial both possible surfaces. pial may be better
        # add dist may or may not affect final results. More testing is needed
        src4 = mne.setup_source_space(subject=sub, spacing='ico4', surface='white', subjects_dir=mridir, add_dist=True, verbose=True)
        src4.save(fname=ico4_path, overwrite=True)
    else:
        src4 = mne.read_source_spaces(fname=ico4_path)
    
    ica_apply = ica_apply.crop(tmin=start, tmax=start+duration)
    epochs = mne.make_fixed_length_epochs(ica_apply, duration=duration, reject_by_annotation=False, preload=True, verbose=True)
    print(epochs)
    print(type(epochs))
    # decimation may not be necessary
    epochs.decimate(decim=10)
    evoked_data = epochs.average()
    # print(evoked_data.nave)
    evoked_data = evoked_data.filter(l_freq=13, h_freq=25, phase='minimum')
    evoked_data.save(fname=evoked_path, overwrite=True)
    
    trans = mne.read_trans(fname=trans_path)
    bem = mne.read_bem_solution(fname=bem_path)
    if (fwd==True or forward_path.exists()==False):
        # cv=3, conflict with nlgc's cv=5?
        forward = mne.make_forward_solution(evoked_data.info, trans=trans, src=src4, bem=bem, meg=True, eeg=False, ignore_ref=True, verbose=True)
        forward.save(fname=forward_path, overwrite=True)
    else:
        forward = mne.read_forward_solution(fname=forward_path, verbose=True)
    
    forward = mne.convert_forward_solution(forward, force_fixed=True)

    
    
    if (covariance==True or cov_path.exists()==False):
        # check method
        empty_causal = empty_causal.crop(tmin=3, tmax=empty.times[-1] - 3)
        cov = mne.compute_raw_covariance(empty_causal, picks='meg', method='auto')
        mne.write_cov(fname=cov_path, cov=cov, overwrite=True)
    else:
        cov = mne.read_cov(fname=cov_path)
    
    if (return_vars == True):
        return evoked_data, forward, cov, src1
    return

server = eelfarm.start_server('localhost')
subject = 'A201'
sessions = ['EC1', 'EO1']
for session in sessions:
    lam = [0.05]
    for trial in range(3):
        try:
            duration = 58
            start_time = 3 + trial * duration
            # evoked, fwd, cov, src = preprocess(sub=subject, session=session, resting=True, empty=True, evoked=True, ico1=True, ico4=True, fwd=True, covariance=True, return_vars=True)
            # if (start_time == 3):
            #     continue
            evoked, fwd, cov, src = preprocess(sub=subject, session=session, start=start_time, duration=duration, return_vars=True)

            dst = f"{result_dir}/beta25hz/4eig2ord/[{subject}]-[visit=]-[session={session}]-[trial={trial}]-[beta]-[testmodel82].p"
            print("nlgc start")
            # server.put(dst, nlgc.nlgc_map, subject, evoked, fwd, cov, src, use_es=False,
            #     order=2, n_eigenmodes=2, patch_idx=list(range(84)),
            #     lambda_range=lam, max_iter=500, max_cyclic_iter=3,
            #     tol=1e-5, sparsity_factor=0.0, var_thr=1.0, cv=5, 
            #     n_orients=1, parallel_process='multiprocess', verbose=True)
            obj = nlgc.nlgc_map(subject, evoked, fwd, cov, src, use_es=False,
                order=2, n_eigenmodes=4, patch_idx=list(range(84)),
                lambda_range=lam, max_iter=500, max_cyclic_iter=3,
                tol=1e-5, sparsity_factor=0.0, var_thr=1.0, cv=5, 
                n_orients=1, parallel_mode='multiprocess', verbose=True)

            eel.save.pickle(obj, dst)


        except Exception as e:
            print(subject, session, trial, " encountered an error ", e)
            continue
print("finished")

