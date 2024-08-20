# rl-safety-filters
Code for Jacques' "RL Safety Filters" mini-project.

The RL algorithms are implemented within the [OmniSafe](https://www.omnisafe.ai/en/latest/) framework. Jacques has made a [fork](https://github.com/JacquesCloete/omnisafe) of the Omnisafe library that adds these algorithms.

## Setup and Installation

```
# Clone this repo
git clone https://github.com/JacquesCloete/rl-safety-filters.git
cd rl-safety-filters

# Create a conda environment from the provided config file
conda env create --file conda-recipe.yaml
conda activate rl-safety-filters

# Clone and install Jacques' fork of omnisafe 
cd ..   # or wherever you want to store the source code -- but NOT in the rl-safety-filters repo!
git clone https://github.com/JacquesCloete/omnisafe.git
cd omnisafe
pip install -e .    # remember to have the conda environment activated before running this!
```