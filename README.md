# ts-eval
Time Series analysis and evaluation tools

## Development

Recommended development workflow:
```
pipenv install -e .[dev]
pipenv shell
```
The library doesn't use Flit/Poetry, so the suggested workflow is based on Pipenv (as per https://github.com/pypa/pipenv/issues/1911).
Pipfile* are ignored in the .gitignore.


**WORK IN PROGRESS**

# TODO:

* metrics:
  * Reference baseline prediction should be optional
  * Baseline prediction using target dataset (without explicit calculation, but losing some time points)
* components
  * refactor components into view/controller, so that computed metrics can be accessed independently
  * Graph: Visualize outliers from confidence interval
  * Graph: plot confint
  * Metrics: builtin slices with holiday
  * Multi-comparison component: scikit_posthocs lib or homecooked?
  * Residual stats: since I have residuals => Ljung-Box, Heteroscedasticity test, Jarque-Bera – like in statsmodels results.
  * inspect true confidence interval coverage via sampling (was done in postings around bayesian dropout sampling)
  * xarrays: compare if compared datasets are actually equal (offets by dates, shapes, maybe even hashing)
* features
  * tests for fourier
  * holiday lib integration (lib should be optional)
  * nint generation
* utils:
  * model adaptor (for different models, generic) which generates 3d prediction dataset. For stastmodels using dyn forecast or kalman filter
  * future importance calculator, but only if I can manipulate input features
  * feature selection using PACF / prewhiten?
* project
  * more definsive style (add arg checks, so it's easier to understand what is going on)
  * docstrings
  * circleci
  * meaningful examples notebook
  * a gif with project visualization
  * badges
  * changelog
  * pypi release

next scope:
* sMAPE & MASE can be added for the jupyter evaluation tables



For multiple comparisons:
    import scikit_posthocs as sp
    sp.posthoc_nemenyi_friedman(pmm)
