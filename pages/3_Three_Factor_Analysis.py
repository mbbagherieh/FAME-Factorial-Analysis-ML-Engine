# FAME — Factorial Analysis & Machine Learning Engine
# Three-Factor Analysis Module

import sys
sys.path.append('..')
from analyzer_core import run_analyzer
run_analyzer(
    n_factors_default=3,
    default_factor_names=["Factor1", "Factor2", "Factor3"],
    default_responses="Response1\nResponse2\nResponse3",
    app_title="Three-Factor Experiment Analyzer"
)
