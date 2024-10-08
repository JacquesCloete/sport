# rl-safety-filters
Code for Jacques' "RL Safety Filters" mini-project.

We plan to eventually implement the RL algorithms within the [OmniSafe](https://www.omnisafe.ai/en/latest/) framework. Jacques has made a [fork](https://github.com/JacquesCloete/omnisafe) of the Omnisafe library for this.

## Setup and Installation

### If Not Using OmniSafe
Just clone this repo into your workspace and install CVXPY in your conda environment (as well as the usuals like NumPy and MatPlotLib).

### If Using OmniSafe
```bash
# Clone this repo
git clone https://github.com/JacquesCloete/rl-safety-filters.git
cd rl-safety-filters

# Create a conda environment from the provided config file
conda env create --file omnisafe-conda-recipe.yaml
conda activate rl-safety-filters

# Clone and install Jacques' fork of omnisafe 
cd ..   # or wherever you want to store the source code -- but NOT in the rl-safety-filters repo!
git clone https://github.com/JacquesCloete/omnisafe.git
cd omnisafe
# Note: DO NOT interact with the conda-recipe in the omnisafe repo
pip install -e .    # remember to have the conda environment activated before running this!
```