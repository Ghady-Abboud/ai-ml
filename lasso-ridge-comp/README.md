# Ridge vs Lasso: Implementation & Experimental Analysis

This project implements **ridge regression** and **lasso regression** from scratch using NumPy and compares their behavior on real and synthetic datasets. The focus is on understanding **regularization**, **optimization**, and **statistical learning** as presented in *Introduction to Statistical Learning (ISL)*. No external ML libraries are used for the core algorithms.

---

## Goals
- Implement ridge, and lasso cleanly from scratch.  
- Study how L1 and L2 regularization affect coefficient shrinkage, sparsity, and error.  
- Produce controlled experiments and clear visualizations that demonstrate these effects.  

---

## Core Components

### **1. Ridge Regression (L2)**

![alt text](images/ridge_equation.png)

Include:
- configurable α  
- MSE evaluation  
- α sweep on a log scale  
- plots of **test error vs α** and **coefficient magnitudes vs α**

---

### **2. Lasso Regression (L1)**
Implemented with **coordinate descent**.

![alt text](images/lasso_equation.png)

> **Reference:** [Coordinate Descent for Lasso Regression](https://xavierbourretsicotte.github.io/lasso_implementation.html#Implementing-coordinate-descent-for-lasso-regression-in-Python)

Include:
- soft-thresholding operator  
- convergence stopping condition  
- warm-start support  
- α sweep  
- plots of **sparsity vs α** and **lasso coefficient paths**

---

### **3. Real Datasets**
Use two small regression datasets (e.g., Boston Housing + one UCI dataset).  
For each dataset:
- ridge across α
- lasso across α
- plots:
  - MSE curves  
  - coefficient shrinkage/sparsity  
  - regularization paths  

---

### **4. High-Dimensional Synthetic Data**
Generate a dataset where:
- features p >> samples n  
- true coefficient vector is sparse  
- Gaussian noise is added  

Evaluate:
- ridge’s stability  
- lasso’s ability to recover the true sparse support  
- plots comparing true vs estimated coefficients
