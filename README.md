# FAME — Factorial Analysis & Machine Learning Engine

A comprehensive, open-source Python pipeline for statistical and machine learning analysis of factorial plant experiments.

## Features
* N-way ANOVA with Type III sums of squares and comprehensive effect sizes (η², ω², ε²)
* Simple effects analysis for significant interactions
* ART ANOVA for design-appropriate non-parametric validation
* Post-hoc mean comparisons (Tukey's HSD / Fisher's LSD) via R's `agricolae` with a native Python fallback
* Benjamini-Hochberg FDR correction (within-response and across-response)
* SHAP feature importance with null-distribution filtering and bootstrap confidence intervals
* Permutation feature importance validation
* GroupKFold cross-validation (to prevent data leakage from replicate samples)
* Model comparison (Random Forest, Linear Regression, Gaussian Process Regression) with Nadeau-Bengio corrected t-tests
* Multi-seed stability analysis
* Multi-Output Random Forest out-of-fold residual correlation analysis
* Principal Component Analysis with biplot visualization
* GPR interaction response surfaces
* Comprehensive simulation study (2×2×2×2 factorial) with known ground truth
* Publication-ready figures (600 DPI PNG + PDF, A4 single/two-column layouts)
* Comprehensive text report generation
* Streamlit web interface (no programming required)

## Installation
To install the required Python dependencies, run:

```
pip install streamlit numpy pandas matplotlib scipy statsmodels scikit-learn shap
text
```

For post-hoc letter display (optional but highly recommended), install R and the `agricolae` package:

R -e "install.packages('agricolae')"
text


## Usage
To launch the Streamlit web interface, run:
```
streamlit run Home.py
text
```

## Validation
The pipeline is validated on two independent biological datasets (*Viola* spp. and *Vicia faba*) and a 4-factor simulation study with known ground truth.

## License
This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

## Citation
If you use FAME in your research, please cite:
*Bagherieh-Najjar MB, Daqeliyeh R, Sadeghipour HR. FAME: An Integrated Factorial Analysis and Machine Learning Pipeline for Plant Experiments. Plant Methods (2026). [Under Review]*
