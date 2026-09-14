import nibabel as nib
import numpy as np
import mne
from scipy.ndimage import gaussian_filter
import subprocess


def make_cortical_hull(sub, mriout, config):
    mgz_path = config.data_src.mridir / sub / "mri/aparc.a2009s+aseg.mgz" 
    output_mask_path = f"{sub}_cortical_hull_mask.mgz"

    all_labels = mne.get_volume_labels_from_aseg(mgz_path)

    # brainstem and cerebellum areas
    exclude_keywords = ["Cerebellum", "Brain-Stem", "4th-Ventricle", "Unknown", "CSF"]

    keep_labels = [
        label for label in all_labels 
        if not any(word in label for word in exclude_keywords)
    ]

    lut, _ = mne.read_freesurfer_lut()

    # map the keeping label strings back to their numeric voxel IDs
    target_ids = [lut[label] for label in keep_labels if label in lut]

    img = nib.load(mgz_path)
    data = img.get_fdata()

    binary_mask = np.isin(data, target_ids)

    # convolve with a 3D Gaussian filter (sigma=2.0 voxels)
    smoothed_data = gaussian_filter(binary_mask.astype(float), sigma=5.0)

    # threshold out the soft edges (0.3 keeps a slightly wider mask boundary)
    final_mask = (smoothed_data > 0.3).astype(np.int16)

    mask_img = nib.Nifti1Image(final_mask, img.affine, img.header)
    nib.save(mask_img, mriout / output_mask_path)

    print(f"Mask successfully created with {len(target_ids)} matched regions.")

    surf_path = f"{sub}_cortical_hull_mask.surf"

    # Construct the FreeSurfer command
    command = ["mri_tessellate", str(mriout / output_mask_path), "1", str(mriout / surf_path)]

    try:
        # Execute the command and capture output
        result = subprocess.run(
            command, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        print("Success! Surface file created successfully.")
        
    except subprocess.CalledProcessError as e:
        print("Error executing mri_tessellate. Make sure FreeSurfer is sourced in your terminal environment.")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise e

