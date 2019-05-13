# community-simulator

## Introduction
This package is designed for simulating batch culture experiments on complex microbial communities. The architecture is based on the standard protocol for these experiments:
- Add media to each of the wells in a 96-well plate. It could be the same media for all wells, or different media for each well, depending on what the experimenter is trying to test.
- Add a small sample of an existing bacterial culture to each of the wells. Again, these could be the same for all wells, or different initial conditions could be tested in parallel.
- Let the bacteria grow while stirring or shaking for some fixed time T.
- Pipette a small fraction of each of the wells into a well on a new plate, with fresh media (added according to the same protocol as in step 1).
- Repeat the previous two steps as many times as desired.

Communities can also be run in chemostat mode, where nutrients are continually supplied and populations continuously diluted.  

## Dependencies
Numpy, Pandas, Matplotlib, SciPy (all included in the standard Anaconda distribution). The `SteadyState` method for finding equilibrium points additional requires CVXPY >= 1.0 (www.cvxpy.org) . 

## Installation
### Mac
To install on a Mac, browse to the community-simulator directory in Terminal, and type
`pip install -e .`
The `-e` flag makes the package 'editable,' so that changes made in the community-simulator directory are carried over to the location where Python stores installed module files. If you are running GitHub Desktop, this allows you to simply hit the "fetch origin" button to update the code to the latest version.

### Windows
The easiest way of installing on Windows is through the Anaconda Navigator (https://www.anaconda.com/download/). Launch the Navigator, and open the Anaconda Terminal. Then navigate to the community-simulator directory, and type
`python -m pip install -e .`

Note that the parallelization features are not currently supported on Windows, so be sure to set `parallel=False` when initializing an instance of the `Community` class.

## Documentation
See the accompanying Jupyter notebook `Tutorial.ipynb` for explanations of the main classes, methods and functions, with illustrative examples. More detailed documentation can be found in the manuscript referenced below.

## Licensing and citation
This package is provided under an MIT license. Please cite us if you use this software: 

Robert Marsland III, Wenping Cui, Joshua Goldford, Pankaj Mehta *The Community Simulator: A Python package for microbial ecology,*  bioRxiv:613836 (2019). 
