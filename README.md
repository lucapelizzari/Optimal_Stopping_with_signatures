# Code for "Primal and dual optimal stopping with signatures"
This repository contains the implementations related to the numerical section of the paper "Primal and dual optimal stopping with signatures" (https://arxiv.org/abs/2312.03444).
The following numerical examples are included:
- Optimal stopping of fractional Brownian motion (lower and upper bounds)
- Pricing American options in rBergomi model (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554754), (lower and upper bounds)

Remarks about the code:
- The dual procedure requires a (free) license from Gurobi optimization, and an explanation how to install it can be found here https://www.gurobi.com/academia/academic-program-and-licenses/. We use Gurobi to efficiently solve high-dimensional linear programs appearing in the dual approach.
- For the simulation of rBergomi model we use (slightly changed version of) the code from R. McCrickerd (https://github.com/ryanmccrickerd/rough_bergomi)
- For the simulation of fractional Brownian motion we use (slightly changed version of) the package C. Flynn (https://pypi.org/project/fbm/)
- For the construction of the signature we use the package iisignature (https://pypi.org/project/iisignature/)
## How to use the code

For both examples we provide notebooks with step by step explanation of how to use the code, by computing lower and upper bounds for a certain choice of parameters, that might be changed by the user. For the sake of reduced running-time, in the notebooks we use smaller sample sizes than described in the numerical section of the paper, which might lead to slightly worse bounds, as the Monte-Carlo error is bigger. Increasing them is possible at the price of increased running time and memory-usage. 
