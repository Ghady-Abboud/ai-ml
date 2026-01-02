# AI/ML Portfolio

This repository documents my journey in Machine Learning and Artificial Intelligence, featuring a comprehensive collection of projects ranging from fundamental implementations to advanced real-world applications. The projects demonstrate proficiency in statistical learning, deep learning, NLP, financial modeling, and algorithm implementation from scratch.

---

## 📁 Repository Structure

### 🎯 Fundamentals
Projects focused on understanding core ML concepts and popular tech stacks.

#### [Breast Cancer Detection](./fundamentals/Breast-Cancer-Detection/)
Deep learning solution for medical image classification using Convolutional Neural Networks.
- **Tech Stack:** TensorFlow, Python, CNN Architecture
- **Highlights:** 
  - Custom CNN with data augmentation pipeline
  - 88% validation accuracy on biopsy image classification
  - Early stopping and learning rate optimization
  - TensorBoard integration for training visualization

#### [Linear Regression](./fundamentals/linearRegression/)
Comprehensive exploration of linear regression fundamentals and implementation.
- **Focus:** Statistical learning basics, gradient descent, model evaluation

#### [ISL Chapter 4 & 5 Labs](./fundamentals/)
Hands-on exercises from *Introduction to Statistical Learning*.
- **Topics:** Classification methods, cross-validation, bootstrap methods
- **Tech Stack:** Python, scikit-learn, statistical modeling

---

### 🚀 Intermediate
Mini-projects demonstrating practical skills with ML libraries and data analysis tools.

#### [Algorithmic Trading Models](./intermediate/algo-trading-models/)
Machine learning trading system for S&P 500 stocks with comprehensive backtesting framework.
- **Tech Stack:** Python, scikit-learn, XGBoost, pandas, yfinance
- **Approach:**
  - Full pipeline: data acquisition → feature engineering → model training → backtesting
  - Technical indicators (RSI, MACD, rolling statistics)
  - Classification models for next-day price direction prediction
- **Results:**
  - Logistic Regression: 52.4% accuracy, 26.25% test return
  - XGBoost: 53% accuracy, 63.92% test return
  - Outperformed naive baseline (46.3%) and random strategy (-0.50%)
- **Note:** Backtesting excludes slippage and transaction costs

---

### 🏆 Main
Advanced projects showcasing deep expertise in ML theory, implementation, and real-world applications.

#### [Geopolitical Tension Forecaster](./main/intelligence-proj/)
Real-time dashboard predicting conflict escalation using NLP and event data analysis.
- **Tech Stack:** HuggingFace (DistilBERT), FastAPI, React, XGBoost, Docker
- **Data Sources:** GDELT Event Database (300M+ events), ACLED conflict data
- **Approach:**
  - Fine-tuned DistilBERT on news headlines for sentiment analysis
  - Combined text embeddings with structured event features
  - 5-level escalation prediction (stable → armed conflict)
  - 30-day ahead forecasting for 8 country pairs
- **Target:** >70% precision/recall on escalation events
- **Status:** In development (data collection & model training phase)

#### [Ridge vs Lasso: Implementation & Analysis](./main/lasso-ridge-comp/)
From-scratch implementation and experimental comparison of regularization techniques.
- **Tech Stack:** NumPy (no ML libraries for core algorithms), matplotlib
- **Implementations:**
  - Ridge Regression (L2 regularization) with closed-form solution
  - Lasso Regression (L1 regularization) via coordinate descent
- **Analysis:**
  - Comprehensive α parameter sweeps
  - Coefficient path visualizations
  - High-dimensional synthetic data experiments (p >> n)
- **Key Findings:**
  - Lasso effectively performs feature selection in sparse models
  - Ridge provides stability with correlated features but retains all features
  - Clear demonstration of L1 vs L2 shrinkage properties

---

## 🛠️ Technologies & Skills

**Languages:** Python  
**ML/DL Frameworks:** TensorFlow, scikit-learn, XGBoost, HuggingFace Transformers  
**Data Science:** pandas, NumPy, matplotlib, statistical analysis  
**NLP:** DistilBERT, text embeddings, sentiment analysis  
**Other:** FastAPI, React, Docker, Git, Jupyter Notebooks

---

## 📚 Learning Resources

This work draws inspiration and knowledge from:
- *Introduction to Statistical Learning* (ISL) - Gareth James, et al.
- GDELT Project & ACLED for geopolitical data
- Various ML research papers and online resources

---

## 🎓 About

This repository represents hands-on learning and project-based skill development in AI/ML, covering:
- **Theory → Practice:** From mathematical derivations to production-ready implementations
- **Diverse Domains:** Healthcare, finance, geopolitics, statistical learning
- **Full-Stack ML:** Data pipelines, model training, evaluation, deployment
- **Research Mindset:** Experimental comparisons, ablation studies, rigorous testing. 
