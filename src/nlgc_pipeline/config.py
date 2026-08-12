from dataclasses import dataclass, field


@dataclass
class PipelineDataSource:
    megdir: str=None
    mridir: str=None
    overwrite: bool=True

@dataclass
class PipelineSystemSpec:
    meg_sys: str = 'KIT'

@dataclass 
class PipelineParticipantScanInfo:
    visit: str = ""
    session: str = 'UNK'
    trials: list=field(default_factory=lambda: [0])
    epoch_duration: int=60
    buffer: int=3 # size of data to be cropped at the beginning and end of MEG recording.

@dataclass
class PipelineFilterParams:
    wideband_lower_bandlimit: float = 1
    wideband_upper_bandlimit: float = 100
    filter_phase: str = 'minimum'
    analysis_lower_bandlimit: float = 1
    analysis_upper_bandlimit: float = 10
    ICA_mode: str = 'auto'
    ICA_components: float = 20
    ICA_method: str = 'infomax'
    sfreq = 250

@dataclass
class PipelineInverseModelSetup:
    source_spaces: list=field(default_factory=list)

@dataclass
class PipelineConfig:
    data_src: PipelineDataSource = field(default_factory=PipelineDataSource)
    system_spec: PipelineSystemSpec = field(default_factory=PipelineSystemSpec)
    scan_info: PipelineParticipantScanInfo = field(default_factory=
                                                   PipelineParticipantScanInfo)
    inverse: PipelineInverseModelSetup = field(default_factory=
                                               PipelineInverseModelSetup)
    filter_params: PipelineFilterParams = field(default_factory=
                                                PipelineFilterParams)
    verbose: bool = True

    # megdir: str=None
    # mridir: str=None

    # evoked cannot be properly saved and thus must be made directly before use
    # meg_type: str='RESTING'
    # session: str='UNK' #just for saving session name in data path
    # overwrite: bool=True
    # epoch_duration: int=60
    # buffer: int=3 # size of data to be cropped at the beginning and end of MEG recording.
    # trials: list=field(default_factory=lambda: [0])
    # make_resting: bool=False
    # make_empty: bool=False
    # make_ICA: bool=False
    # ICA_mode: str='auto'
    # components: float=30
    # source_spaces: list=field(default_factory=list)
    # make_forward: bool=False
    # make_covariance: bool=False





    

