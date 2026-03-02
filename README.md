# Deep Birth-Death

This repository accompanies the manuscript:

Peña P.G., Iglesias G., Talavera E., Meseguer A.S. & Sanmartin I. (2025). *On the utility of Deep Learning for model classification and parameter estimation on complex diversification scenarios*.

## What You Will Find Here

- Deep learning and maximum likelihood workflows for phylogenetic diversification model classification and parameter estimation.
- Precomputed simulated datasets for multiple tree sizes (`87`, `489`, `674` tips).
- Pretrained deep learning classifiers and regressors (`.keras` + training metadata).
- MLE inference outputs for TreePar and DDD on simulated and empirical trees.
- Jupyter notebooks for dataset generation, training, benchmarking, and empirical prediction.
- Source modules under `src/` used by notebooks and scripts.

## Repository Layout

- `deep_learning/`: deep learning notebooks, models, serialized datasets, calibration temperatures, and revision experiments.
- `MLE/`: TreePar and DDD scripts, notebooks, and saved inference outputs.
- `simulations/`: simulation scripts and generated datasets used in DL and MLE workflows.
- `src/`: reusable Python/R utilities for encoding, dataset loading, evaluation, and MLE helpers.
- `empirical/`: empirical phylogenies and confidence-interval figures.
- `envs/`: conda/micromamba environment stubs.

## Quick Start

```bash
git clone https://github.com/pablogpena/deep_birth_death.git
cd deep_birth_death
```

## Typical Workflows

### 1) Use pretrained DL models for empirical prediction

- Main notebook: `deep_learning/Empirical_Phylogenies_Prediction.ipynb`
- Required assets are in `deep_learning/models/`, `deep_learning/temperatures/`, and `empirical/*.nwk`.

### 2) Rebuild datasets and train DL models

Suggested notebook order:

1. `deep_learning/Generate_Raw_Data.ipynb`
2. `deep_learning/Generate_Dataset.ipynb`
3. `deep_learning/Train_Models.ipynb`
4. `deep_learning/Results_Classification_Simulations.ipynb`
5. `deep_learning/Results_Regression_Simulations.ipynb`
6. `deep_learning/Results_Classification_Treepar_Data.ipynb`
7. `deep_learning/Results_Regression_Treepar_Data.ipynb`

### 3) Run MLE inference (TreePar and DDD)

Example commands:

```bash
cd MLE/DDD
bash run_DDD.sh /absolute/path/to/deep_birth_death/simulations/treepar_dataset/674
bash run_DDD_empiric.sh

cd ../TreePar
bash run_TreePar.sh /absolute/path/to/deep_birth_death/simulations/treepar_dataset/674
bash run_TreePar_empiric.sh
```

Results are saved under `MLE/inference_data/`.

Associated analysis notebooks:

- `MLE/classification_results.ipynb`
- `MLE/regression_results.ipynb`
- `MLE/empiric_results.ipynb`

### 4) Simulate new phylogenies

Simulation scripts are in `simulations/code/`:

- `sim_phylogeny.r`
- `sim_DD.r`
- `simulate_all.sh`
- configuration files in `simulations/config_sim.r` and `simulations/config_sim_real_rho.r`

Example (simulate all scenarios):

```bash
cd simulations/code

# Simulate BD, HE, ME, SR and WW
bash simulate_all.sh BD HE ME SR WW

# Simulate DD scenarios
Rscript sim_DD.r
```

## Contact

- Pablo G. Peña: `pgutierrez@rjb.csic.es`


## Preprint

Peña P.G., Iglesias G., Talavera E., Meseguer A.S. & Sanmartin I. (2025). *On the utility of Deep Learning for model classification and parameter estimation on complex diversification scenarios*.

bioRxiv: https://www.biorxiv.org/content/10.1101/2025.08.27.671290v2
  
