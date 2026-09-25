# Antagonist review

## Stop pretending a warm start branch is a persistence feature

The original wrapper recreated the searcher on every `fit()` despite the core having a warm start branch.
That discarded useful work whenever a user tried to extend a search through the public interface.
The fitted core also fails an ordinary `joblib` save because metric wrappers are closures.
Expecting users to discover those details and choose a serializer themselves was a poor lifecycle contract.

The public selector now provides explicit save/load/resume methods, retains the original training inputs and metric, checks versions, and recreates workers on demand.
Fresh-process tests establish that saved closures, selected features, and prior search history survive loading and that resumed searches add evaluations.
The remaining architectural debt is the generated core's ownership: a Colab source link without a local generator is an invitation to divergence.
An adversary would put the generating source and reproducible export command under version control, then move search lifecycle responsibilities into maintained modules with explicit boundaries.
Keep the new public API small and test behavior through it instead of making users manipulate the internal searcher.

## Count contracts, not green checks

The first 32 passing tests missed a constructor parameter being overwritten and an entire search phase bypassing the advertised strategy allocator.
That is what happens when a validation count substitutes for a map from each product promise to an observable invariant.
The stronger workflow traces allocation through every phase and exercises fit, resume, save, load, clone, and refit as one lifecycle.
Keep scheduling in a maintained module with one shared phase implementation instead of accumulating another special-purpose tail loop in generated source.
