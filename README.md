# sport

[![Python: 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper [SPoRt - Safe Policy Ratio: Certified Training and Deployment of Task Policies in Model-Free RL](https://arxiv.org/abs/2504.06386) (IJCAI'25).

## Setup and Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/JacquesCloete/sport.git
    ```

1. Create a `sport` conda environment from the provided config file:
    ```bash
    cd sport
    conda env create --file conda_envs/sport.yaml
    conda activate sport
    cd ..
    ```

1. Clone and install Jacques' forks of [Gymnasium](https://github.com/JacquesCloete/Gymnasium) and [Safety Gymnasium](https://github.com/JacquesCloete/safety-gymnasium) (remember to have the sport conda environment activated before doing this!):
    ```bash
    git clone https://github.com/JacquesCloete/Gymnasium.git
    cd Gymnasium
    git checkout jacques/v0.28.1
    pip install .
    cd ..

    git clone https://github.com/JacquesCloete/safety-gymnasium.git
    cd safety-gymnasium
    git checkout jacques/v1.2.0
    pip install .
    cd ..
    ```

1. Install this project (again, remember to have the `sport` conda environment activated before doing this!):
    ```bash
    cd sport
    pip install -e .
    ```

## Experiments

I use Hydra and W&B to configure and log experiments. Trained policies and collected scenario databases from a run are saved in the corresponding run folder, which can be found in the `src/sport/outputs` folder. Run folders are labelled by date and time of start.

Make sure that when running experiments that use policies generated from earlier steps, you are correctly using them in later steps! (Check experiment script configs, and in particular, the directories and file names that are searched to get the policy weights). Also when plotting, make sure you've copied all the required data from the experiment runs into the right directories.

## Running Experiments
Navigate to `src/sport` (`cd src/sport`) to run experiments. You should run experiments from that directory.

### Experiment 1 (Pre-Trained Task Policy)

1. Train base policy: `python sac_safety_train.py`

2. Scenario-based validation of base policy:
`python sac_safety_validate.py`
(depending on your CPU/memory you may need to reduce `validate_common.num_envs`; I suggest starting low and increasing until your CPU or memory usage is close to maxed out)

3. Train task policy (*without* maintaining a bound on failure probability): `python sac_safety_train.py --config-name=sac_safety_unsafe`

4. Collect performance data for the projected policy over different alphas:
`wandb sweep --project sport config/projected_ppo_validate_sweep_fixed_env.yaml`

### Experiment 2 (Task Policy Trained Using Projected PPO)

1. Train base policy: `python sac_safety_train.py`

2. Scenario-based validation of base policy:
`python sac_safety_validate.py`
(depending on your CPU/memory you may need to reduce `num_envs`; I suggest starting low and increasing until your CPU or memory usage is close to maxed out)

3. Train task policy while maintaining a bound on failure probability: `wandb sweep --project sport config/projected_ppo_finetune_sweep_fixed_env.yaml`

4. Collect performance data for the projected policy over different alphas (using the task policy trained at the same alpha):
`wandb sweep --project sport config/projected_ppo_validate_sweep_fixed_env_use_alpha_task.yaml`

## Plotting Experiments
Notebooks can be found in `notebooks/envs`.

Plot graphs of performance data for the projected policy over different alphas: `comparing_validation_over_alphas_low_freq_fixed_env.ipynb`

Plot episode trajectory distributions for the projected policy over different alphas: `episode_trajectory_distribution_visualization.ipynb`

Plot policy projections over an episode trajectory: `plotting_policy_projection.ipynb` (requires first running `extract_frames_for_interpreting_policy_projection.ipynb`)

## Citation
If you find this code useful in your research, please consider citing our paper:
```bibtex
@inproceedings{sport,
    title     = {{SPoRt} - {Safe Policy Ratio}: Certified Training and Deployment of Task Policies in Model-Free {RL}},
    author    = {Cloete, Jacques and Vertovec, Nikolaus and Abate, Alessandro},
    booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
    publisher = {International Joint Conferences on Artificial Intelligence Organization},
    year      = {2025}
}
```

## License
This project is licensed under the terms of the [MIT License](LICENSE).