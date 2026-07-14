# FLAVORS2 Agent Instructions

## Product direction

FLAVORS2 is the feature-selection analogue of FLAML's cost-efficient search optimization.
Preserve stochastic global and local exploration as a core product property.
Do not turn the selector into a deterministic recursive-elimination algorithm.

Structured signals such as coefficients, feature importances, priors, and ranked subsets may propose search candidates.
They must not permanently eliminate features or constrain the search to one nested path.

Allocate evaluation budget according to expected improvement and evaluation cost.
Maintain enough global exploration, uncertainty sampling, and population diversity to escape local optima.
Prefer adaptive portfolios of search strategies over one fixed proposal mechanism.

When optimizing benchmark performance, compare selectors under shared data splits, estimators, metrics, and compute budgets.
Validate changes across multiple datasets and seeds rather than tuning to one holdout result.
