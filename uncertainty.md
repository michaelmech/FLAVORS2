# Uncertainty log

## Fitted search persistence

- **Uncertainty:** Can a fitted selector be saved and resumed across processes?
- **Why it matters:** A failed or partial search may need more budget later without discarding its evaluated subsets and ECI history.
- **Initial hypothesis and confidence:** The core warm start branch suggested this might work through ordinary `joblib` persistence; low confidence because no save/load API or serialization test exists.
- **Consequence if wrong:** Users may believe a saved wrapper resumes when it actually starts a fresh search, or encounter a serialization error.
- **Evidence:** `src/flavors2/core.py` initializes a new core object in wrapper `fit()` and wraps metrics in closures during core initialization.
- **Resolution:** A small fitted-object check showed `joblib.dump()` fails on the wrapped default metric.
  `cloudpickle` saved and restored the core object, and the restored core entered the warm start branch on a second `fit()`.
  The wrapper discarded its restored core object on `fit()`.
  The public selector now implements `save()`, `load()`, and `resume(budget=...)` in `src/flavors2/selector.py`.
  Checkpoints include private copies of training inputs and version metadata.
  Resume accepts only additional budget and uses the retained data and fitted metric, avoiding incompatible replacement inputs.
  Round-trip tests verify transform, selection, RNG, ECI, cache, and population preservation, and a fresh-process test verifies additional evaluations after loading a local metric closure.

## Generated core ownership

- **Uncertainty:** Where should the persistence API live when `core.py` is marked as automatically generated?
- **Evidence:** The header identifies a Colab export; no generating source notebook or export workflow is present in this checkout.
- **Hypothesis and confidence:** A separate public lifecycle module can extend the generated wrapper without editing its source; high confidence.
- **Consequence if wrong:** Direct changes to the generated file could be lost on regeneration or violate repository instructions.
- **Resolution:** Implemented the public subclass in `src/flavors2/selector.py` and exported it from the package root.
  The maintained `src/flavors2/search.py` subclass owns phase scheduling while inheriting evaluation and ECI primitives from the generated core.

## Review findings: refinement allocation and resume parameters

- **Uncertainty:** Do the public lifecycle and final search phase preserve the documented constructor budget and fair-chance strategy allocation?
- **Evidence:** PR #2 comments 4101646148 and 4101646156 identified a local-only refinement pool and assignment to the wrapper budget inside resume.
- **Hypothesis and confidence:** Both violate the public contract despite the initial suite passing; high confidence after reproducing them through public fits.
- **Consequence if wrong:** Later fits silently receive the incremental budget, or a fixed search phase spends budget without consulting strategy ECI.
- **Resolution:** Regression tests failed before the fixes and passed afterward.
  The maintained search subclass in src/flavors2/search.py routes all three phases through the adaptive portfolio and is used by both package-root public selectors.
  Resume passes its budget only to the retained searcher.
  Tests cover the final phase across three seeds and both public entry points, plus checkpoint, clone, and refit budget preservation.

## Search allocator chronology

- **Uncertainty:** Should the post-selection size override or the inverse-ECI allocator determine actual proposal strategies?
- **Evidence:** Commit d12983b6 introduced the override on July 13, while 4d49c717 introduced the fair-chance allocator on July 14; e7b56387 had omitted timeout cost on July 10.
- **Hypothesis and confidence:** The newer ECI allocator is authoritative; high confidence from this chronology and the stated search contract.
- **Consequence if wrong:** Local and uncertain proposals receive less than their fair share, while costly timeouts fail to influence ECI.
- **Resolution:** The maintained search path keeps stochastic size exploration inside the global and ranked portfolio strategies and records real elapsed timeout cost without a successful score.
