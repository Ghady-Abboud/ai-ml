# Ridge vs Lasso: Implementation & Experimental Analysis

This project implements **ridge regression** and **lasso regression** from scratch using NumPy and compares their behavior on real and synthetic datasets. The focus is on understanding **regularization**, **optimization**, and **statistical learning** as presented in *Introduction to Statistical Learning (ISL)*. No external ML libraries are used for the core algorithms.

---

## Goals
- Implement OLS, ridge, and lasso cleanly from scratch.  
- Study how L1 and L2 regularization affect coefficient shrinkage, sparsity, and error.  
- Produce controlled experiments and clear visualizations that demonstrate these effects.  
- Show understanding of statistical learning concepts at a solid undergraduate level.

---

## Core Components

### **1. Linear Regression (Baseline)**
- Closed-form OLS  
- Gradient descent version  
- MSE metric  
- Prediction function  
Used for baseline comparisons.

---

### **2. Ridge Regression (L2)**
Closed-form:

\[
\hat\beta = (X^T X + \lambda I)^{-1} X^T y
\]

Include:
- configurable λ  
- coefficient shrinkage tracking  
- MSE evaluation  
- λ sweep on a log scale  
- plots of **test error vs λ** and **coefficient magnitudes vs λ**

---

### **3. Lasso Regression (L1)**
Implemented with **coordinate descent**, following ISL.

Include:
- soft-thresholding operator  
- convergence stopping condition  
- warm-start support  
- λ sweep  
- plots of **sparsity vs λ** and **lasso coefficient paths**

This is the most important part of the project.

---

### **4. Regularization Paths**
Generate full regularization paths for both ridge and lasso:
- For each λ, fit model and store coefficients.  
- Plot **coefficient values vs λ** for both methods.  
These visualizations clearly show the difference between L1 and L2 behavior.

---

### **5. Real Datasets**
Use two small regression datasets (e.g., Boston Housing + one UCI dataset).  
For each dataset:
- baseline OLS performance  
- ridge across λ  
- lasso across λ  
- plots:
  - MSE curves  
  - coefficient shrinkage/sparsity  
  - regularization paths  

---

### **6. High-Dimensional Synthetic Data**
Generate a dataset where:
- features p >> samples n  
- true coefficient vector is sparse  
- Gaussian noise is added  

Evaluate:
- ridge’s stability  
- lasso’s ability to recover the true sparse support  
- plots comparing true vs estimated coefficients
