# Project 1 - Time Series

## Setting up environment

### Requirements

### Setup

To create the environment run:

    'conda env create -f environment.yml --prefix venv'

To activate the environment run: 

    'conda activate venv -n ml4hc_p1'

To deactivate the environment run:

    'conda deactivate'

To update the environment run:

    'conda env update --prefix ./venv --file environment.yml  --prune'

## Running models

Always run the models from the root directory (where the README.md resides)

### RNN

Set the model in the train_rnn.py file at the bottom in the function call. Then run:

    python3 src/train/train_rnn.py

## Folder Structure 
```
├── README.md          <- The top-level README
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, model summaries (evaluation results), etc.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-ts-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── src                <- Source code for use in this project.
    ├── data           <- Scripts to download, generate, and process data
    ├── models         <- Scripts defining the models
    ├── train          <- Scripts for training the models
    └── visualization  <- Scripts rendering (static) visualizations for reports etc.
```
The project folder structure is based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/).

