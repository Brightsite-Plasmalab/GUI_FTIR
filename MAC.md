# Installation

Install [Homebrew](https://brew.sh/) if you don't have it already.  
Install `c-blosc` and `HDF5`:  `brew install c-blosc hdf5`
Install a virtual environment: `python3 -m venv .venv` (Python 3.14.2 tested on MacOS Sequoia)
Activate the virtual environment: `source .venv/bin/activate`
Install the package: `pip install -e .`
Run the GUI: `python -m Code_Gui.FTIR_GUI`