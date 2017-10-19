# community-simulator

## Introduction
This package is designed for simulating batch culture experiments on complex microbial communities. The architecture is based on the standard protocol for these experiments, which we can roughly summarize as follows:
- Add media to each of the wells in a 96-well plate. It could be the same media for all wells, or different media for each well, depending on what the experimenter is trying to test.
- Add a small sample of an existing bacterial culture to each of the wells. Again, these could be the same for all wells, or different initial conditions could be tested in parallel.
- Let the bacteria grow while stirring or shaking for some fixed time T, usually chosen to be close to the time when the optical density stops increasing.
- Pipette a small fraction of each of the wells into a well on a new plate, with fresh media (added according to the same protocol as in step 1).
- Repeat the previous two steps as many times as desired.

This simulator is designed to replicate this kind of experiment, and to facilitate visualization and analysis of the results.
