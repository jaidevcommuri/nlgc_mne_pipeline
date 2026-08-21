Install by downloading and `pip install -e .`

Usage:
```
import nlgc_pipeline as npl
import nlgc_pipeline.preprocess as pipeline
import pathlib


config = npl.config.PipelineConfig()
config.data_src.megdir = pathlib.Path("./meg/")
config.data_src.mridir = pathlib.Path("./mri/")
config.scan_info.session = "CocktailSZ"
sub = "RXXXX"
verbose = True


tsss_causal, ica = pipeline.fit_filters(sub, config, verbose)

ica_apply = pipeline.apply_ica(sub, config, verbose=True)

empty = pipeline.filter_empty(sub, config, verbose=True)

for space in ["vol20", "vol5"]:
    pipeline.make_src(sub, config, space=space, verbose=verbose)

# by default, will extract a 60 second trial from start of data
evoked = pipeline.make_evoked(sub, config, trial=0, verbose=True)

# forward may in the future depend on ICA fit per trial, hence one forward per trial
pipeline.make_fwd(sub, config, evoked=evoked, trial=0, space="vol5", verbose=True)

cov = pipeline.make_cov(sub, config, empty, verbose=verbose)

```
