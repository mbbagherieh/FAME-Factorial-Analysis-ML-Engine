# FAME — Factorial Analysis & Machine Learning Engine
# Two-Factor Analysis Module
import sys
sys.path.append('..')
from analyzer_core import run_analyzer

run_analyzer(
    n_factors_default=2,
    default_factor_names=["WaterDeficit_pct", "Chitosan_pct"],
    default_responses="FreshWeight\nDryWeight\nChla_b_ratio\nChlt\nCAT\nPOD\nProtein",
    app_title="Two-Factor Experiment Analyzer"
)
