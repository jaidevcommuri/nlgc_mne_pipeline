import warnings
from pathlib import Path
import mne
import pymeshfix
import numpy as np
from mne.io.constants import FIFF


def make_bem(sub, mriout, config):
    mne.bem.make_watershed_bem(
        subject=sub,
        subjects_dir=config.data_src.mridir,
        overwrite=True,
    )
    
    bem_path = Path(
        f"{mriout}/{sub}-inner_skull-bem-sol.fif"
    )

    model = mne.make_bem_model(
        subject=sub,
        ico=4,
        conductivity=(0.3, 0.006, 0.3),
        subjects_dir=config.data_src.mridir,
    )

    bem = mne.make_bem_solution(model)
    mne.write_bem_solution(bem_path, bem, overwrite=True)


def make_scalp_surfaces(subject, subjects_dir):
    """Create scalp surfaces, repairing the dense surface if needed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        print(subject, subjects_dir)
        mne.bem.make_scalp_surfaces(
            subject,
            subjects_dir=subjects_dir,
            overwrite=True,
        )

    if any("topological defects" in str(w.message) for w in caught):
        repair_scalp_surface(subject, subjects_dir)


def repair_scalp_surface(subject, subjects_dir):
    bem_dir = Path(subjects_dir) / subject / "bem"
    for density in ["dense", "medium", "sparse"]:
        fname = bem_dir / f"{subject}-head-{density}.fif" 

        # ignore since we are going to fix them right now
        surf = mne.read_bem_surfaces(fname, on_defects='ignore')[0]

        meshfix = pymeshfix.MeshFix(surf["rr"], surf["tris"])
        meshfix.repair()

        mne.write_head_bem(
            fname,
            meshfix.points,
            meshfix.faces,
            overwrite=True,
        )


def _remove_implausible_dig_pts(raw, coreg):
    omit_dist_mm = 15 # mm away from scalp surface
    coreg.omit_head_shape_points(distance=omit_dist_mm / 1000) # in meters

    dig = raw.info['dig']

    good_extra_indices = list(np.where(coreg._extra_points_filter)[0])

    dig_clean = [
        d for d in dig
        if d["kind"] != FIFF.FIFFV_POINT_EXTRA
        or d["ident"] in good_extra_indices
    ]

    # see https://mne.tools/mne-bids/stable/_modules/mne_bids/dig.html
    # this is how mne adds dig pts
    with raw.info._unlock():
        raw.info["dig"] = dig_clean

    return raw, coreg