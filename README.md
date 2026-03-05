# AI/ML Projects

A collection of machine learning projects I've worked on over the past 3-4 months in my spare time.

---

## fundamentals/linearRegression

Implements linear regression from scratch using gradient descent, then compares the result against scikit-learn's implementation. Uses a synthetic dataset of study hours vs. exam scores.

## fundamentals/ISL-Chapter4-Lab

Classification lab from Introduction to Statistical Learning, Chapter 4. Applies logistic regression, LDA, QDA, Naive Bayes, and KNN to predict S&P 500 daily direction (up/down) using the Smarket dataset (2001–2005).

## fundamentals/ISL-Chapter5-Lab

Resampling lab from Introduction to Statistical Learning, Chapter 5. Uses the validation set approach with polynomial regression (degrees 1–3) to predict MPG from horsepower on the Auto dataset.

## fundamentals/Breast-Cancer-Detection

CNN binary classifier that distinguishes malignant from benign breast tissue images. Uses a custom architecture with batch normalization, dropout, and data augmentation. Achieves ~88% validation accuracy on a 70/20/10 train/val/test split.

---

## intermediate/algo-trading-models

End-to-end algorithmic trading pipeline for S&P 500 stocks. Downloads daily OHLCV data from Yahoo Finance, engineers features (RSI, MACD, lag returns), and trains logistic regression and XGBoost models to predict next-day price direction. Includes backtesting with a simulated equity curve. XGBoost returned +63.9% over the 2023–2025 test period (before slippage/fees).

---

## main/lasso-ridge-comp

NumPy-only implementations of Ridge (closed-form) and Lasso (coordinate descent) regression. Compares L1 vs. L2 regularization on synthetic high-dimensional data (p >> n) and the diabetes dataset. Results confirm Lasso outperforms Ridge when the true model is sparse.

## main/intelligence-proj

Geopolitical conflict forecasting that attempts to predict Iran-Israel escalation 30 days ahead. Pulls GDELT event data and ACLED conflict labels, scores event severity using BERT zero-shot classification, and aggregates features over rolling 30-day windows. Shelved as a feasibility study — the ~180M BERT inference calls required are computationally prohibitive without cloud GPU access.

---

## hyperTune

Bayesian optimization engine written in Rust. Implements a Gaussian Process with RBF kernel and Expected Improvement acquisition function to minimize black-box functions. Demonstrated on the Rosenbrock function over 30 iterations. No external ML dependencies — built from first principles using nalgebra and statrs.
