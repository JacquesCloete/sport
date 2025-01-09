# rl_vcf
**R**einforcement **L**earning with scenario-based **V**alidation and projection-based **C**onstrained **F**inetuning.

## Setup and Installation

```bash
# Clone this repo
git clone https://github.com/JacquesCloete/rl-vcf.git
cd rl-vcf

# Create a conda environment from the provided config file
conda env create --file conda_envs/rl_vcf.yaml
conda activate rl_vcf

# Clone and install Jacques' forks of Gymnasium and Safety Gymnasium
# (remember to have the rl_vcf conda environment activated before doing this!)
cd ..
git clone https://github.com/JacquesCloete/Gymnasium
cd Gymnasium
git checkout jacques/v0.28.1
pip install .

cd ..
git clone https://github.com/JacquesCloete/safety-gymnasium
cd safety-gymnasium
git checkout jacques/v1.2.0
pip install .

# Install this project
# (remember to have the rl_vcf conda environment activated before doing this!)
cd ..
cd rl-vcf
pip install -e .
```

## Experiments

I use Hydra and W&B to configure and log experiments. Trained policies and collected scenario databases from a run are saved in the corresponding run folder, which can be found in the `src/rl_vcf/outputs` folder. Run folders are labelled by date and time of start.

Make sure that when running experiments that use policies and scenario databases generated from earlier steps, you are correctly selecting them in later steps! (Check input args)

## Running Experiments
Navigate to `cd src/rl_vcf` to run experiments.

### Experiment 1

1. Train base policy: `python sac_safety_train.py train.per=True train.curriculum=True train_common.gym_id=SafetyPointReachAvoidCurriculum-v0`

2. Scenario-based validation of base policy:
`python sac_safety_validate.py`
(depending on your CPU/memory you may need to reduce `validate_common.num_envs`; I suggest starting low and increasing until your CPU or memory usage is close to maxed out)

3. Train task policy (*without* maintaining a bound on failure probability): `python sac_safety_train.py train=sac_safetypointreachavoid_unsafe_low_freq safety_common=sac_safetypointreachavoid_unsafe_low_freq`

4. Collect performance data for the projected policy over different alphas:
`wandb sweep --project rl_vcf config/projected_ppo_validate_sweep_fixed_env.yaml`

### Experiment 2

1. Train base policy: `python sac_safety_train.py train.per=True train.curriculum=True train_common.gym_id=SafetyPointReachAvoidCurriculum-v0`

2. Scenario-based validation of base policy:
`python sac_safety_validate.py`
(depending on your CPU/memory you may need to reduce `num_envs`; I suggest starting low and increasing until your CPU or memory usage is close to maxed out)

3. Train task policy while maintaining a bound on failure probability: `python projected_ppo_finetune.py` (choose desired alpha by setting `train.alpha`)

4. Collect performance data for the projected policy over different alphas:
`wandb sweep --project rl_vcf config/projected_ppo_validate_sweep_fixed_env.yaml` (remember to use the task policy you trained in the previous step!)

## Plotting Experiments
Notebooks can be found in `notebooks/envs`.

### Experiment 1
Plot graphs of performance data for the projected policy over different alphas: `comparing_validation_over_alphas_low_freq_fixed_env.ipynb`