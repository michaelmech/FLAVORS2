<p align="center">
  <img src="https://raw.githubusercontent.com/michaelmech/FLAVORS2/main/assets/pareto_graph.png" width="720" />
</p>

---

[![PyPI](https://img.shields.io/pypi/v/flavors-squared)](https://pypi.org/project/flavors-squared/)
![Python](https://img.shields.io/pypi/pyversions/flavors-squared)
![License](https://img.shields.io/github/license/michaelmech/FLAVORS2)
![Downloads](https://img.shields.io/pypi/dm/flavors-squared)
[![Build](https://github.com/michaelmech/FLAVORS2/actions/workflows/publish.yml/badge.svg)](https://github.com/michaelmech/FLAVORS2/actions)

---

<h2 align="center">FLAVORS² — Adaptive Feature Selection with Pareto Search</h2>

<p align="center">
  <b>Fast and Lightweight Assessment of Variables for Optimally-Reduced Subsets towards Feature Learning Automation with Variable-Objective, Resource Scheduling.</b>
</p>

---

FLAVORS² is a flexible metaheuristic framework for feature subset selection in supervised learning.  
It supports **multi-objective optimization**, **custom metrics**, **time-bounded search**, **feature priors**, and **importance-aware weighting**.

---

## 🧩 Install

Install from PyPI (note the package name):

<pre>
pip install flavors-squared
</pre>

Then import:

```python
from flavors2 import FLAVORS2FeatureSelector
