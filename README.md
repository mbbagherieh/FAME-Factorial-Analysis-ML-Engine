# FAME — Factorial Analysis & Machine Learning Engine

A comprehensive, open-source Python pipeline for statistical and machine learning analysis of factorial plant experiments.

## Features

- N-way ANOVA with effect sizes (η²)
- Kruskal-Wallis non-parametric validation
- Post-hoc mean comparisons (Tukey's HSD / Fisher's LSD) via R's agricolae
- SHAP feature importance with permutation validation
- Random Forest, Linear Regression, and Gaussian Process Regression with cross-validation
- Bootstrap confidence intervals for standardized regression coefficients
- Multi-Output Random Forest residual correlation analysis
- Principal Component Analysis with biplot visualization
- GPR interaction response surfaces
- Publication-ready figures (600 DPI PNG + PDF, A4 single/two-column layouts)
- Comprehensive text report generation
- Streamlit web interface (no programming required)

## Installation

```bash
pip install streamlit numpy pandas matplotlib scipy statsmodels scikit-learn shap
```
For post-hoc letter display (optional):
```bash
# Install R and agricolae package
R -e 'install.packages("agricolae")'
```
## Usage
```bash
streamlit run Home.py
```
License

MIT License
Citation

[Citation to be added after publication]

