# Framework edge cases

## 2026-09-24T22:40:59-07:00 - Fitted metric wrappers and persistence

The generated core wraps every metric in nested Python functions.
Even the default static metric becomes a closure that ordinary pickle and `joblib.dump()` cannot serialize by its original qualified name.
The symptom is a `PicklingError` stating that the wrapped default metric is not the same object as `flavors2.core.FLAVORS2.default_metric`.
Use the public selector's cloudpickle-based `save()` and `load()` methods, which also check checkpoint and environment versions.
Local custom metrics must be exercised by loading in a fresh Python process; a same-process test can conceal dependencies on notebook or test globals.
This was easy to miss because sklearn compatibility suggests ordinary estimator persistence, while successful loky worker execution already uses a serializer capable of handling closures.

## 2026-09-24T22:40:59-07:00 - Resuming with too little time for an initial evaluation

The generated core's initial candidate timeout and insufficient-capacity paths append a fallback subset and set `best_error` to infinity, including on a warm start with an existing finite incumbent.
The public `resume()` method preserves the earlier best error if a new phase regresses that value, then synchronizes selection and ECI attributes from the core.
The regression test forces insufficient evaluation capacity after loading a successful search and checks that both the prior selection and best error survive.
This was difficult to detect through ordinary fit tests because infinity is a valid fallback for a fresh search with no successful evaluations.
Only a second timed phase with preexisting search history exposes the broken assumption.

## 2026-09-24T23:20:03-07:00 - Incremental budgets and sklearn constructor parameters

The persistence wrapper originally assigned resume's incremental duration to self.budget.
Sklearn get_params and clone read that same constructor attribute, so an apparently successful warm start silently changed the next fresh fit and every later clone.
Keep the operation budget on the retained searcher and preserve wrapper constructor parameters.
The original tests checked history growth and invalid budgets, which concealed the missing round-trip lifecycle invariant.
A regression now checks resume, get_params, clone, checkpoint loading, and a new fit together.
