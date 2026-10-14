# FAME — Factorial Analysis & Machine Learning Engine
# Two-Factor Analysis Module
import sys
sys.path.append('..')
from analyzer_core import run_analyzer

run_analyzer(
    n_factors_default=2,
    default_factor_names=["WaterDeficit_pct", "Chitosan_pct"],
    default_responses="Chlorophyll\nFreshWeight\nDryWeight\nCAT\nPOD\nProtein\nProline",
    app_title="Two-Factor Experiment Analyzer"
)
