import mne
import pathlib
import eelfarm
import nlgc
import numpy
import eelbrain as eel
import numpy as np
from dataclasses import dataclass, field

@dataclass
class pipeline_config:
    megdir: str=None
    mridir: str=None

    # evoked cannot be properly saved and thus must be made directly before use
    meg_sys: str='KIT'
    meg_type: str='RESTING'
    session: str='UNK' #just for saving session name in data path
    overwrite: bool=True
    epoch_duration: int=60
    buffer: int=3 # size of data to be cropped at the beginning and end of MEG recording.
    trials: list=field(default_factory=lambda: [0])
    make_resting: bool=False
    make_empty: bool=False
    make_ICA: bool=False
    ICA_mode: str='auto'
    components: float=30
    source_spaces: list=field(default_factory=list)
    make_forward: bool=False
    make_covariance: bool=False



    

