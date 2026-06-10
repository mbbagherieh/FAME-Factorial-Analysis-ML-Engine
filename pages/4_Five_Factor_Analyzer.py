# FAME — Factorial Analysis & Machine Learning Engine
# Five-Factor Analysis Module
import sys
sys.path.append('..')
from analyzer_core import run_analyzer
run_analyzer(
    n_factors_default=5,
    default_factor_names=["Factor1", "Factor2", "Factor3", "Factor4", "Factor5"],
    default_responses="Response1\nResponse2\nResponse3",
    app_title="Five-Factor Experiment Analyzer"
)
