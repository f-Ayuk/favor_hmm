# favor_hmm

## Project layout

- `activity_hmm.py` is the command-line entry point.
- `pipeline.py` coordinates dataset preparation, training, evaluation, and plots.
- `config.py` contains activity labels and processing settings.
- `data_processing.py` loads, resamples, and splits sensor recordings.
- `feature_extraction.py` builds time- and frequency-domain window features.
- `hmm_model.py` trains and saves the Gaussian Hidden Markov Model.
- `evaluation.py` calculates metrics and writes predictions.
- `visualization.py` creates the result figures.

## Run

Install the packages listed in `requirements.txt`, then run from the project
folder:

```powershell
python activity_hmm.py
```

Generated reports, model files, and figures are written to `results/`.