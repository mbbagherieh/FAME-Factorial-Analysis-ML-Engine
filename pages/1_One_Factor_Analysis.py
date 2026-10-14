# ============================================================
# FAME — Factorial Analysis & Machine Learning Engine
# One-Factor Analysis Module
# ============================================================

import sys
sys.path.append('..')
from analyzer_core import run_analyzer

run_analyzer(
    n_factors_default=1,
    default_factor_names=["Treatment"],
    default_responses="Response1\nResponse2\nResponse3",
    app_title="One-Factor Experiment Analyzer"
)
