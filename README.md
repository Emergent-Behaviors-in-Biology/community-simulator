# community-simulator

## Introduction
This package is designed for simulating batch culture experiments on complex microbial communities. The architecture is based on the standard protocol for these experiments, which we can roughly summarize as follows:
- Add media to each of the wells in a 96-well plate. It could be the same media for all wells, or different media for each well, depending on what the experimenter is trying to test.
- Add a small sample of an existing bacterial culture to each of the wells. Again, these could be the same for all wells, or different initial conditions could be tested in parallel.
- Let the bacteria grow while stirring or shaking for some fixed time T, usually chosen to be close to the time when the optical density stops increasing.
- Pipette a small fraction of each of the wells into a well on a new plate, with fresh media (added according to the same protocol as in step 1).
- Repeat the previous two steps as many times as desired.

This simulator is designed to replicate this kind of experiment, and to facilitate visualization and analysis of the results.

## Installation
To install on a Mac, browse to the community-simulator directory in Terminal, and type
`pip install -e .`
The `-e` flag makes the package 'editable,' so that changes made in the community-simulator directory are carried over to the location where Python stores installed module files. If you are running GitHub Desktop, this allows you to simply hit the "fetch origin" button to update the code to the latest version.
