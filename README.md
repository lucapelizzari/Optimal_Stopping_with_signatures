# Optimal_Stopping_with_signatures
Implementation of primal and dual method for the optimal stopping problem, see https://arxiv.org/abs/2312.03444 for details.

Examples include: 
- Optimal stopping of fractional Brownian motion (lower and upper bounds)
- Pricing American options in rBergomi model (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554754), (lower and upper bounds)
- Pricing American options in rHeston model (https://onlinelibrary.wiley.com/doi/abs/10.1111/mafi.12173), (lower bounds, upper bounds under construction)

Remarks:
- The dual procedure requires a (free) license from Gurobi optimization (https://www.gurobi.com/). The latter we use to efficiently solve high-dimensional linear programs.
- For the simulation of rHeston model, we use the code of S. Breneis (https://github.com/SimonBreneis/approximations_to_fractional_stochastic_volterra_equations/tree/master)
- The simulation of rBergomi model we use (slightly changed version of) the code from R. McCrickerd (https://github.com/ryanmccrickerd/rough_bergomi)
- For the simulation of fractional Brownian motion we use (slightly changed version of) the package C. Flynn (https://pypi.org/project/fbm/)
- For the construction of the signature we use the package iisignature (https://pypi.org/project/iisignature/)
