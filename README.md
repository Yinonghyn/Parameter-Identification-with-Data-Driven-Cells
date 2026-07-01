# Parameter-Identification-with-Data-Driven-Cells

This repository contains Python code based on our work *Data-Adaptive Learning of Dynamical Systems by Matching Transfer Operators and Invariant Measures*.

- `weight_Lorenz63.ipynb`: This file contains visualizations of weights and modified Ulam matrices.
- `Synthetic Experiments`: This folder contains the codes of our synthetic experiments on clean and noisy Lorenz-63 and Lorenz-96 systems.
  - `Noise-free Benchmark`: Illustration of the effectiveness of our method in a clean environment. 
  - `Comparison with Pointwise and SINDy`: Visual comparison of our method with the pointwise method and SINDy in a noisy environment.
  - `Error Computation`: Numerical comparison of our method and the pointwise method.
- `Sea Surface Temperature`: This folder contains the application of our method to the NOAA SST Dataset.
  - `SST_Matrix_Measure_Comparison.py`: This file predicts the SST evolution using the Markov matrix matching and invariant measure matching objectives.
  #- `SST-test.ipynb`: This notebook contains POD time series prediction, prediction of our method given the noisy POD coefficients, and comparison with the pointwise method.
