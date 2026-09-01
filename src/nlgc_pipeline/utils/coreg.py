from mne.coreg import Coregistration
from nlgc_pipeline.utils.surfaces import (make_scalp_surfaces, 
                                          _remove_implausible_dig_pts)
import pathlib
import numpy as np
import mne


def compute_coreg(sub, config, raw, megout, verbose=False): 
    trans_path = pathlib.Path(
        f"{megout}/{sub}-trans.fif"
    )

    # coreg already done (or manually redone)
    if trans_path.exists():
        return raw
    
    # need scalp surfaces for coreg
    print("subjects dir = ", config.data_src.mridir)
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