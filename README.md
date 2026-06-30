# Parameter-Identification-with-Data-Driven-Cells

This repository contains Python code based on our work *Data-Adaptive Meshes for Learning Dynamical Systems: Matching Transition Matrices and Invariant Measures*.

- `weight_Lorenz63.ipynb`: This file contains visualizations of weights and modified Ulam matrices.
- `Synthetic Experiments`: This folder contains the codes of our synthetic experiments on clean and noisy Lorenz-63 and Lorenz-96 systems.
  - `Noise-free Benchmark`: Illustration of the effectiveness of our method in a clean environment. 
  - `Comparison with Pointwise and SINDy`: Visual comparison of our method with the pointwise method and SINDy in a noisy environment.
  - `Error Computation`: Numerical comparison of our method and the pointwise method.
- `Sea Surface Temperature`: This folder contains the application of our method to the NOAA SST Dataset.
  - `SST-test.ipynb`: This notebook contains POD time series prediction, prediction of our method given the noisy POD coefficients, and comparison with the pointwise method.


####The following illustrates the evolution of the sea surface temperature predicted with our method and the pointwise method compared with the ground truth.
![comparison](https://github.com/user-attachments/assets/07163d69-cb96-45ba-8ba1-701cdb7fded6)


The following illustrates the absolute errors over time of our method and the pointwise method.
![Absolute Difference](https://github.com/user-attachments/assets/63546bd2-030d-430e-870d-96b2e56c6138)
