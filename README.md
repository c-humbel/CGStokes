# A Conjugate Gradient-based solver for the full Stokes equations running on GPUs

This repository contains the work done in my master's thesis project. The goal is to implement a conjugent gradient method that solves the full Stokes equation, which arises in simulations of ice flow dynamics. The algorithm will be designed specifically for execution on GPUs using the Julia programming language.

The folder structure of the project is the following:

- **scripts**: scripts documenting the development progress of the solver
-  **experiments**: numerical experiments used in the final report
- **src**: the most final version of the kernels, which are used to perform the experiments
- **meetings**: short documents that summarise the montly progress meetings
- **jacobian_construction**: scripts that were used to check properties of Jacobians at different stages of the development
- **variationa_consistency**:  scripts that were used to check variational consistency in a simplified setup
- **data**:  input data for the experiments
- **figure**: figures used in the progress description 