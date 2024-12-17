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