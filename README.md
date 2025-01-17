# rl_vcf
**SPoRt** - **S**afe **Po**licy **R**a**t**io : Certified Training and Deployment of Task Policies in Model-Free RL.

(Repo originally named **R**einforcement **L**earning with scenario-based **V**alidation and projection-based **C**onstrained **F**inetuning, hence **rl-vcf**.)

## Setup and Installation

### Standard (Non-Anynomous) Repo
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

### Anonymous Repo for Reviewers
```bash
# Download this repo: https://anonymous.4open.science/r/rl-vcf-3C26/
cd rl-vcf   # Enter rl-vcf repo folder

# Create a conda environment from the provided config file
conda env create --file conda_envs/rl_vcf.yaml  # Create conda env
conda activate rl_vcf   # Activate conda env

# Clone and install Jacques' forks of Gymnasium and Safety Gymnasium
# (remember to have the rl_vcf conda environment activated before doing this!)
cd .. # Exit repo folder
# Download this repo: https://anonymous.4open.science/r/Gymnasium-51B2/
cd Gymnasium    # Enter Gymnasium repo folder
pip install .   # Install Gymnasium

cd .. # Exit Gymnasium repo folder
# Download this repo: https://anonymous.4open.science/r/safety-gymnasium-7E5D/
cd safety-gymnasium # Enter Safety Gymnasium repo folder
pip install .   # Install Safety Gymnasium

# Install this project
# (remember to have the rl_vcf conda environment activated before doing this!)
cd ..   # Exit Safety Gymnasium repo folder
cd rl-vcf   # Enter rl-vcf repo folder
pip install -e .    # Install rl-vcf
```


## Experiments

I use Hydra and W&B to configure and log experiments. Trained policies and collected scenario databases from a run are saved in the corresponding run folder, which can be found in the `src/rl_vcf/outputs` folder. Run folders are labelled by date and time of start.

Make sure that when running experiments that use policies generated from earlier steps, you are correctly using them in later steps! (Check experiment script configs, and in particular, the directories and file names that are searched to get the policy weights). Also when plotting, make sure you've copied all the required data from the experiment runs into the right directories.

## Running Experiments
Navigate to `cd src/rl_vcf` to run experiments. You should run experiments from that directory.

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

3. Train task policy while maintaining a bound on failure probability: `wandb sweep --project rl_vcf config/projected_ppo_finetune_sweep_fixed_env.yaml`

4. Collect performance data for the projected policy over different alphas (using the task policy trained at the same alpha):
`wandb sweep --project rl_vcf config/projected_ppo_validate_sweep_fixed_env_use_alpha_task.yaml`

## Plotting Experiments
Notebooks can be found in `notebooks/envs`.

Plot graphs of performance data for the projected policy over different alphas: `comparing_validation_over_alphas_low_freq_fixed_env.ipynb`

Plot episode trajectory distributions for the projected policy over different alphas: `episode_trajectory_distribution_visualization.ipynb`

Plot policy projections over an episode trajectory: `plotting_policy_projection.ipynb` (requires first running `extract_frames_for_interpreting_policy_projection.ipynb`)