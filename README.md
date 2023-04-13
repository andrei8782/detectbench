# detectbench
A benchmarking and optimisation framework for spike detection algorithms supported by SpikeInterface.


`run.ipynb` represents a demo pipeline showcasing the currently available functionality.

Usage guide:
1. Install `miniconda` (https://docs.conda.io/en/latest/miniconda.html)
2. Recreate the conda environment: `conda env create -f environment.yml`
3. Activate the environment: `conda activate detectbench`
4. Try out `run.ipynb`
5. Develop your own experiments.

This framework heavily relies on `SpikeInterface` (https://github.com/SpikeInterface). For development purposes, installing it from source is recommended. Follow the instructions here: https://spikeinterface.readthedocs.io/en/latest/installation.html#from-source.

Note: Also install `neo` and `probeinterface` from source (instructions are in the link above) as they are used by `spikeinterface`.

Although `Python 3.11` is supported by `spikeinterface` and is used in the environment file, some spike detection methods require the use of `numba` which is not yet compatible with `Python 3.11`. For this reason, installing `Python 3.10` will be necessary until the dependencies are updated to support `3.11`.
