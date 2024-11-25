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

# Install this project
pip install -e .    # remember to have the conda environment activated before running this!
```