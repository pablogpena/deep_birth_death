# Deep Birth-Death: Deep Learning for Model Classification & Parameter Estimation

This repository accompanies the manuscript *"On the utility of Deep Learning for model classification and parameter estimation on complex diversification scenarios."*  
It contains scripts, notebooks, data, and models to reproduce and extend the analyses presented in the paper.

## Table of Contents
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Contact](#contact)
- [Preprint](#Preprint)

---

## Installation

Clone the repository:
   ```bash
   git clone https://github.com/pablogpena/deep_birth_death.git
   cd deep_birth_death
```
## Repository Structure 

- **/deep_learning/** – All features for Deep Learning:
    - Notebooks for data generation, model training/testing, and empirical inference.
    - **models/** – Trained classification and regression DL models, and training data.
    - **pickles/** – Generated simulation datasets for model training and testing.
    - **revision_test/** – Experiments performed after the first manuscript revision.
    - **temperatures/** – Data used to scale softmax probabilities.
- **/MLE/** – All features for Maximum Likelihood Estimation:
    - Notebooks for model selection, parameter inference, and empirical analyses using MLE.
    - **DDD/** – Scripts for running DDD on simulated and empirical trees.
    - **TreePar/** – Scripts for running TreePar on simulated and empirical trees.
    - **inference_data/** – Results of MLE inference.
    - inference_example.txt – Instructions for running DDD and TreePar. 
- **/envs/** – Conda environments for performing MLE inference:
    - `treepar.yml` – Environment for TreePar.
    - `r_env.yml` – Environment for DDD.
- **/simulations/** – Jupyter notebooks for exploratory analysis and visualization.
    - Code to define parameter simulation bounds and notebooks for rescaling them.
    - **treepar_dataset/** – Dataset used for MLE.
    - **simulated_trees/** – Dataset used for DL.
    - **code/** – Scripts for simulation with TreeSim.
- **/empirical/** – Empirical datasets: phylogenetic trees of eucalypts, conifers, and cetaceans.
- **/src/** – Functions for creating datasets, training/testing DL models, performing MLE inference, and evaluating DL and MLE results.

  ## Contact

For questions or suggestions regarding this repository, please contact:

- **Pablo G. Peña** – Email: pgutierrez@rjb.csic.es
- **Guillermo Iglesias** - Email: guillermo.iglesias@upm.es
- GitHubs: [https://github.com/pablogpena](https://github.com/pablogpena), [https://github.com/guillermoih](https://github.com/guillermoih)

  ## Preprint
  Peña P.G., Iglesias G., Talavera E., Meseguer A.S. & Sanmartín I. (2025) *On the utility of Deep Learning for model classification and parameter estimation on complex diversification scenarios*. [bioRxiv](https://www.biorxiv.org/cgi/content/short/2025.08.27.671290v1)
  
