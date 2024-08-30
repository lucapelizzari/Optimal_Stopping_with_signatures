# Code for "Primal and dual optimal stopping with signatures"
This repository contains the implementations related to the numerical section of the paper "Primal and dual optimal stopping with signatures" (https://arxiv.org/abs/2312.03444), as well as extended methods relying on deep and kernel learning methodologies, accompanying a working paper on "American option pricing in rough volatility models".

## How to use the code

A step-by-step guidance with notebooks is provided for:
- Optimal stopping of fractional Brownian motion (lower and upper bounds) (Example_Optimal_Stopping_FBM.ipynb)
- Pricing American options in rBergomi model (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554754), (lower and upper bounds) (Example_American_Put_Option_rBergomi.ipynb)

Additionally, for American options in the rough Bergomi we provide two files Testing_linear_signature_stopping.py and Testing_deep_signature_stopping.py, where one can play around with different (hyper) parameters and model choices. Notice that the implementation does not depend on the underlying model, and these examples can easily be modified for different models by simply changing the simulation of the training and testing data.

## Remarks about the code:
- The module Signature_computer.py relies in the package iisignature (https://pypi.org/project/iisignature/), and allows to compute log and standard signatures for various variation of underlying paths.
- The LinearDualSolver in Linear_signature_optimal_stopping.py has the option of choosing Gurobi optimization to solve the linear programs, which requires a free license (an explanation how to install it can be found here https://www.gurobi.com/academia/academic-program-and-licenses/). It is recommended to use it for high-dimensional problems, but alternatively one can set LP_solver ="CVXPY", to use the free cvxpy solvers.
- For the simulation of rBergomi model we use (slightly changed version of) the code from R. McCrickerd (https://github.com/ryanmccrickerd/rough_bergomi)
- For the simulation of fractional Brownian motion we use (slightly changed version of) the package C. Flynn (https://pypi.org/project/fbm/)

