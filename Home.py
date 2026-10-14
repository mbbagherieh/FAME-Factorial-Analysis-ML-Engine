# ============================================================
# Home.py — Landing Page for Plant Stress Analyzer Suite
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="Plant Stress Experiment Analyzer",
    page_icon="🌿",
    layout="wide"
)

# Lab header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=120)
with col2:
    st.markdown("""
    **Plant Stress Physiology Laboratory**  
    Department of Biology, Faculty of Science  
    Golestan University, Gorgan, Iran  
    📧 mb.bagherieh@gu.ac.ir; m.b.bagherieh@gmail.com
    """)

st.divider()

# ============================================================
# WELCOME PAGE
# ============================================================
st.title("🌿 FAME — Factorial Analysis & Machine Learning Engine")
st.caption("Plant Stress Experiment Analyzer Suite")
st.markdown("""
Welcome to the Plant Stress Physiology Laboratory's data analysis platform.  
Select the appropriate analysis based on your experimental design.
""")

st.markdown("---")

    
# ============================================================
# ANALYSIS CARDS
# ============================================================
st.header("📊 Select Your Experimental Design")

# --- Row 1: One-Factor & Two-Factor ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔬 One-Factor Analysis
    **Simplest design:**  
    e.g., Control vs Treatment, or multiple levels of one factor
    
    - 1 independent variable (factor) with 2+ levels
    - Any number of response variables
    - One-way ANOVA
    - Post-hoc mean comparisons (Tukey's HSD / Fisher's LSD)
    - SHAP feature importance
    - Model comparison (RF, LR, GPR)
    
    ➡️ **Use for:** Dose-response, single treatment effects,  
    time-point comparisons (single factor)
    """)
    
    if st.button("🚀 Launch One-Factor Analysis", use_container_width=True, type="primary", key="btn_1f"):
        st.switch_page("pages/1_One_Factor_Analysis.py")

with col2:
    st.markdown("""
    ### 🔬 Two-Factor Analysis
    **Most common design:**  
    e.g., Water Deficit × Chitosan
    
    - 2 independent variables (factors)
    - Any number of response variables
    - Two-way ANOVA with interaction
    - Post-hoc mean comparisons (Tukey's HSD / Fisher's LSD)
    - SHAP feature importance
    - GPR interaction surfaces
    
    ➡️ **Use for:** Water Deficit × Elicitor,  
    Salinity × Hormone, Temperature × Nutrient
    """)
    
    if st.button("🚀 Launch Two-Factor Analysis", use_container_width=True, type="primary", key="btn_2f"):
        st.switch_page("pages/2_Two_Factor_Analysis.py")

st.markdown("---")

# --- Row 2: Three-Factor & Four-Factor ---
col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    ### 🔬 Three-Factor Analysis
    **Complex design:**  
    e.g., Water Deficit × Chitosan × Time
    
    - 3 independent variables (factors)
    - Any number of response variables
    - Three-way ANOVA with all 2-way interactions
    - Post-hoc mean comparisons (Tukey's HSD / Fisher's LSD)
    - SHAP feature importance
    - Pairwise GPR interaction surfaces
    
    ➡️ **Use for:** Multi-stress interactions,  
    Time-course experiments with multiple treatments
    """)
    
    if st.button("🚀 Launch Three-Factor Analysis", use_container_width=True, type="primary", key="btn_3f"):
        st.switch_page("pages/3_Three_Factor_Analysis.py")

with col4:
    st.markdown("""
    ### 🔬 Four-Factor Analysis
    **Advanced design:**  
    e.g., Water × Chitosan × Light × Nutrient
    
    - 4 independent variables (factors)
    - Any number of response variables
    - Multi-way ANOVA with all 2-way interactions
    - Post-hoc mean comparisons (Tukey's HSD / Fisher's LSD)
    - SHAP feature importance
    - Pairwise GPR interaction surfaces
    
    ➡️ **Use for:** Complex factorial experiments,  
    Multi-stress physiology studies
    """)
    
    if st.button("🚀 Launch Four-Factor Analysis", use_container_width=True, type="primary",  key="btn_4f"):
        st.switch_page("pages/4_Four_Factor_Analysis.py")

st.markdown("---")

# --- Row 3: Five-Factor (centered in first column) ---
col5, col6 = st.columns(2)

with col5:
    st.markdown("""
    ### 🔬 Five-Factor Analysis
    **High-dimensional design:**  
    e.g., Full factorial screening
    
    - 5 independent variables (factors)
    - Any number of response variables
    - Multi-way ANOVA with all 2-way interactions
    - Post-hoc mean comparisons (Tukey's HSD / Fisher's LSD)
    - SHAP feature importance
    - Pairwise GPR interaction surfaces
    
    ➡️ **Use for:** Exploratory screening,  
    Large factorial designs
    """)
    
    if st.button("🚀 Launch Five-Factor Analysis", use_container_width=True,  type="primary", key="btn_5f"):
        st.switch_page("pages/5_Five_Factor_Analysis.py")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
### 📋 Before You Begin

1. **Prepare your CSV file** with factor columns and response columns
2. **Know your column names** exactly as they appear in the CSV
3. **Select the analysis** matching your number of factors
4. **Upload your data** on the analysis page

### ❓ Need Help?

Contact Dr. Bagherieh-Najjar: mb.bagherieh@gu.ac.ir; m.b.bagherieh@gmail.com
""")
