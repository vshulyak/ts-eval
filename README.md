# ```ts-eval``` Time Series analysis and evaluation tools

[![pypi](https://img.shields.io/pypi/v/ts-eval)](https://pypi.org/project/ts-eval/)
[![Build Status](https://travis-ci.org/vshulyak/ts-eval.svg?branch=master)](https://travis-ci.org/vshulyak/ts-eval)
[![codecov](https://codecov.io/github/vshulyak/ts-eval/branch/master/graph/badge.svg)](https://codecov.io/github/vshulyak/ts-eval)
[![python3](https://img.shields.io/pypi/pyversions/ts-eval)](https://www.python.org/downloads/release/python-374/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/pypi/l/ts-eval)](https://github.com/vshulyak/ts-eval/blob/master/LICENSE)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/vshulyak/ts-eval/issues)

---
A set of tools to help you analyse time series using Python.

## 🧩 Current features

* **N-step ahead time series evaluation** – using a Jupyter widget.
* **Friedman / Nemenyi rank test (posthoc)** – to see which model statistically performs better.
* **Relative Metrics** – rMSE, rMAE + Forecasted Value analogues.
* **Prediction Interval Metrics** – MIS, rMIS, FVrMIS
* **Fixed fourier series generation**  – fixed in time according to pandas index
* **Naive/Seasonal models for baseline predictions** (with prediction intervals)
* **Statsmodels n-step evaluation** – helper functions to evaluate n-step ahead forecasts using Statsmodels models or naive/seasonal naive models.

Here's how the Jupyter widget looks like:
![Demo Screenshot](images/demo_screenshot.png)

Check out [Demo Notebook](https://nbviewer.jupyter.org/github/vshulyak/ts-eval/blob/master/examples/basic_usage.ipynb).

## Installation

      pip install ts-eval


## 📋 Release Planning:

* Release 0.3
  * travis: add 3.8 default python when it's available
  * docs: supported metrics & API options
  * Maybe use api like Summary in statsmodels MLEModel class, it has extend methods and warn/info messages
  * pretty legend for lots like here https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/
  * Look for TODOs
  * changable colors
  * turn off colored display option
  * a nicer API for raw metrics container
  * codacy badge
  * coverage badge
  * travis/circleci badge
  * violin plots to compare predictions – areas can be colored, different metrics on left and right (like relative...)
* Release 0.4
  * names of datasets passed in (now default ordinal numbers)
  * holiday/fourier features model
  * fix viz module to have less of important stuff
  * a gif with project visualization
  * check shapes of input arrays (target vs preds), now it doesn't raise an error
  * Baseline prediction using target dataset (without explicit calculation, but losing some time points)
  * Graph: plot confint


## 💡 Ideas

* components
  * Graph: Visualize outliers from confidence interval
  * Multi-comparison component: scikit_posthocs lib or homecooked?
  * inspect true confidence interval coverage via sampling (was done in postings around bayesian dropout sampling)
  * xarrays: compare if compared datasets are actually equal (offets by dates, shapes, maybe even hashing)
  * bin together step performance, like steps 0-1, 2-5, 6-12, 13-24
  * highlight regions using a mask (holidays, etc.)
  * option to view interactively points using widget (plotly)?
  * diagnostics: bias to over / underestimate points
  * animated graphs for change in seasonality
* features
  * example notebook for fourier?
  * tests for fourier
  * nint generation
* utils:
  * model adaptor (for different models, generic) which generates 3d prediction dataset. For stastmodels using dyn forecast or kalman filter
  * future importance calculator, but only if I can manipulate input features
  * feature selection using PACF / prewhiten?
* project
  * more defensive style (add arg checks, so it's easier to understand what is going on)
  * docstrings
  * circleci
  * https://timothycrosley.github.io/portray/ for docs
* sMAPE & MASE can be added for the jupyter evaluation tables
* ? Residual stats: since I have residuals => Ljung-Box, Heteroscedasticity test, Jarque-Bera – like in statsmodels results,
  but probably these stats were inspected already by the user... and on which step should they be computed then?


## 🤹🏼‍♂️ Development

Recommended development workflow:
```
pipenv install -e .[dev]
pipenv shell
```
The library doesn't use Flit/Poetry, so the suggested workflow is based on Pipenv (as per https://github.com/pypa/pipenv/issues/1911).
Pipfile* are ignored in the .gitignore.
