from dataclasses import dataclass, field, asdict
import pathlib
import json
from nlgc_pipeline import checksum

@dataclass
class PipelineDataSource:
    megdir: str=None
    mridir: str=None
    outdir: str=None
    config_save: str=None
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
    sfreq: int = 250

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

    metadata: checksum.PipelineNetwork = field(default_factory=checksum.PipelineNetwork)
    
    verbose: bool = True

    def save(self, sub, name=None):
        if name is None:
            name = "config.json"
        
        dst = f"{self.data_src.config_save}/{sub}"
        pathlib.Path(dst).mkdir(parents=True, exist_ok=True)

        with open(f"{dst}/{name}", "w") as f:
            json.dump(asdict(self), f, indent=4)






    

