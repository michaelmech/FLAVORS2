# FLAVORS2 Agent Instructions

## Product direction

FLAVORS2 is the feature-selection analogue of FLAML's cost-efficient search optimization.
Estimated Cost for Improvement (ECI) is the organizing principle of the search, not an optional heuristic or disconnected helper.
Evaluation budget allocation must be driven primarily by dynamically updated ECI estimates derived from observed improvement and evaluation cost.
Candidate subsets must also be compared using their estimated evaluation cost relative to their improvement potential.
Do not introduce ECI-named methods that are absent from the active candidate-generation and selection path.

Preserve stochastic global and local exploration as a core product property.
Do not turn the selector into a deterministic recursive-elimination algorithm.

Structured signals such as coefficients, feature importances, priors, and ranked subsets may propose search candidates.
They must not permanently eliminate features or constrain the search to one nested path.

Allocate evaluation budget according to expected improvement and evaluation cost.
Sample proposal strategies primarily in proportion to inverse ECI, while retaining an explicit fair-chance probability for every strategy.
Keep ECI direction-safe by using the selector's internal lower-is-better error representation.
Expose enough ECI history and diagnostics to verify that estimates respond to successful and unsuccessful trials.
Maintain enough global exploration, uncertainty sampling, and population diversity to escape local optima.
Prefer adaptive portfolios of search strategies over one fixed proposal mechanism.

When optimizing benchmark performance, compare selectors under shared data splits, estimators, metrics, and compute budgets.
Validate changes across multiple datasets and seeds rather than tuning to one holdout result.
