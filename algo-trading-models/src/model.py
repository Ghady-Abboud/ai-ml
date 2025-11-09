import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

class Model:
  def __init__(self, df : pd.DataFrame):
    self.df = df
    self.X_train = None
    self.X_test = None
    self.y_train = None
    self.y_test = None

  def create_labels(self):
    self.df["TARGET_RETURN"] = self.df.groupby("Ticker")["Close"].pct_change().shift(-1)
    self.df["TARGET_DIRECTION"] = (self.df["TARGET_RETURN"] > 0).astype(int)
    self.df.dropna(inplace=True)
    print(self.df.columns)

  def logRegression(self):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(self.X_train, self.y_train)
    return model
  
  def plot_xgb_learning_curve(self, params=None):
    """
    Plot learning curves from xgb.cv to visualize overfitting and optimal trees.
    This helps you see if your model is underfitting or overfitting.
    """
    if params is None:
      params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.2,
      }

    cv_results = xgb.cv(
      params,
      xgb.DMatrix(self.X_train, label=self.y_train),
      num_boost_round=500,
      nfold=5,
      metrics="logloss",
      early_stopping_rounds=30,
      verbose_eval=False,
    )

    # Plot train vs test error
    plt.figure(figsize=(10, 6))
    plt.plot(cv_results.index, cv_results['train-logloss-mean'], label='Train', linewidth=2)
    plt.plot(cv_results.index, cv_results['test-logloss-mean'], label='Test', linewidth=2)
    plt.fill_between(cv_results.index,
                     cv_results['test-logloss-mean'] - cv_results['test-logloss-std'],
                     cv_results['test-logloss-mean'] + cv_results['test-logloss-std'],
                     alpha=0.2)

    optimal_trees = len(cv_results)
    best_score = cv_results['test-logloss-mean'].iloc[-1]

    plt.axvline(optimal_trees, color='red', linestyle='--', label=f'Optimal: {optimal_trees} trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Log Loss')
    plt.title(f'XGBoost Learning Curve\nOptimal Trees: {optimal_trees}, Best LogLoss: {best_score:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('xgb_learning_curve.png', dpi=150)
    plt.close()

    print(f"Optimal number of trees: {optimal_trees}")
    print(f"Best test logloss: {best_score:.4f}")
    print(f"Learning curve saved to: xgb_learning_curve.png")

    return optimal_trees, cv_results

  def tune_xgboost_params(self, quick=True):
    """
    Systematically tune XGBoost hyperparameters and plot results.

    Args:
      quick: If True, uses smaller grid (faster). If False, uses comprehensive grid.

    Returns:
      Best parameters found
    """
    print("\n=== XGBoost Parameter Tuning ===")

    if quick:
      # Quick grid for initial exploration
      max_depths = [3, 5, 7, 10]
      learning_rates = [0.01, 0.05, 0.1]
      colsample_bytrees = [0.8]
    else:
      # Comprehensive grid
      max_depths = [3, 5, 7, 10, 15]
      learning_rates = [0.01, 0.05, 0.1, 0.2]
      colsample_bytrees = [0.6, 0.8, 1.0]

    results = []

    total_combos = len(max_depths) * len(learning_rates) * len(colsample_bytrees)
    combo_num = 0

    for max_depth in max_depths:
      for lr in learning_rates:
        for colsample in colsample_bytrees:
          combo_num += 1

          params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': lr,
            'max_depth': max_depth,
            'colsample_bytree': colsample,
            'seed': 42,
          }

          # Run CV
          cv_results = xgb.cv(
            params,
            xgb.DMatrix(self.X_train, label=self.y_train),
            num_boost_round=500,
            nfold=5,
            metrics="logloss",
            early_stopping_rounds=30,
            verbose_eval=False,
          )

          optimal_trees = len(cv_results)
          best_score = cv_results['test-logloss-mean'].iloc[-1]

          results.append({
            'max_depth': max_depth,
            'learning_rate': lr,
            'colsample_bytree': colsample,
            'n_estimators': optimal_trees,
            'logloss': best_score,
          })

          if combo_num % 5 == 0:
            print(f"  Progress: {combo_num}/{total_combos} - depth={max_depth}, lr={lr}, score={best_score:.4f}")

    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('logloss')

    print("\n=== Top 10 Parameter Combinations ===")
    print(results_df.head(10).to_string(index=False))

    # Plot results
    self._plot_tuning_results(results_df)

    best_params = results_df.iloc[0].to_dict()
    print(f"\n=== Best Parameters ===")
    print(f"max_depth: {int(best_params['max_depth'])}")
    print(f"learning_rate: {best_params['learning_rate']}")
    print(f"colsample_bytree: {best_params['colsample_bytree']}")
    print(f"n_estimators: {int(best_params['n_estimators'])}")
    print(f"Best logloss: {best_params['logloss']:.4f}")

    # Save results
    results_df.to_csv('xgb_tuning_results.csv', index=False)
    print(f"\nFull results saved to: xgb_tuning_results.csv")

    return best_params

  def _plot_tuning_results(self, results_df):
    """Plot parameter tuning results to visualize relationships."""
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Max depth vs LogLoss
    depth_scores = results_df.groupby('max_depth')['logloss'].agg(['mean', 'min'])
    axes[0, 0].plot(depth_scores.index, depth_scores['mean'], 'o-', label='Mean', linewidth=2)
    axes[0, 0].plot(depth_scores.index, depth_scores['min'], 's-', label='Best', linewidth=2)
    axes[0, 0].set_xlabel('Max Depth')
    axes[0, 0].set_ylabel('Log Loss')
    axes[0, 0].set_title('Impact of Max Depth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Learning rate vs LogLoss
    lr_scores = results_df.groupby('learning_rate')['logloss'].agg(['mean', 'min'])
    axes[0, 1].plot(lr_scores.index, lr_scores['mean'], 'o-', label='Mean', linewidth=2)
    axes[0, 1].plot(lr_scores.index, lr_scores['min'], 's-', label='Best', linewidth=2)
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('Log Loss')
    axes[0, 1].set_title('Impact of Learning Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')

    # 4. Heatmap: depth vs learning_rate
    pivot = results_df.pivot_table(values='logloss', index='max_depth', columns='learning_rate', aggfunc='min')
    im = axes[1, 1].imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    axes[1, 1].set_xticks(range(len(pivot.columns)))
    axes[1, 1].set_yticks(range(len(pivot.index)))
    axes[1, 1].set_xticklabels([f'{x:.2f}' for x in pivot.columns])
    axes[1, 1].set_yticklabels(pivot.index)
    axes[1, 1].set_xlabel('Learning Rate')
    axes[1, 1].set_ylabel('Max Depth')
    axes[1, 1].set_title('LogLoss Heatmap (Lower is Better)')
    plt.colorbar(im, ax=axes[1, 1])

    # Add text annotations
    for i in range(len(pivot.index)):
      for j in range(len(pivot.columns)):
        text = axes[1, 1].text(j, i, f'{pivot.values[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig('xgb_tuning_plots.png', dpi=150)
    plt.close()
    print(f"Tuning plots saved to: xgb_tuning_plots.png")

  def xgboost(self):
    model = xgb.XGBClassifier(
      objective='binary:logistic',
      eval_metric='logloss',
      learning_rate=0.05,
      max_depth=3,
      colsample_bytree=0.8,
      n_estimators=118,
    )
    model.fit(self.X_train, self.y_train)
    return model

  def run(self):
    """
    Perform time series cross-validation to evaluate model performance.

    This trains 30 models (one per fold) and averages their performance.
    This is STANDARD practice in ML to get robust performance estimates.

    Yes, it trains 30 models, but:
    - Each model trains in ~5-10 seconds (Logistic Regression is fast)
    - Total time: ~5-10 minutes (acceptable for research)
    - Result: Honest, robust performance metric
    """

    print("\n=== Time Series Cross-Validation ===")

    exclude_cols = ['Date', 'Ticker', 'Close', 'Open', 'High', 'Low', 'Volume', 'TARGET_RETURN', 'TARGET_DIRECTION']
    feature_cols = [col for col in self.df.columns if col not in exclude_cols]

    self.create_labels()    
    X = self.df[feature_cols].values
    y = self.df['TARGET_DIRECTION'].values

    cv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=20)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []

    for fold_num, (train_index, test_index) in enumerate(cv.split(X), 1):

      self.X_train, self.X_test = X[train_index], X[test_index]
      self.y_train, self.y_test = y[train_index], y[test_index]

      # y_pred = self.logRegression().predict(self.X_test)
      y_pred = self.xgboost().predict(self.X_test)
      acc = accuracy_score(self.y_test, y_pred)
      prec = precision_score(self.y_test, y_pred, zero_division=0)
      rec = recall_score(self.y_test, y_pred, zero_division=0)

      fold_accuracies.append(acc)
      fold_precisions.append(prec)
      fold_recalls.append(rec)

      if fold_num % 10 == 0:
        print(f"  Fold {fold_num}/30 - Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

    avg_accuracy = np.mean(fold_accuracies)
    avg_precision = np.mean(fold_precisions)
    avg_recall = np.mean(fold_recalls)

    print("\n=== Cross-Validation Results ===")
    print(f"Average Accuracy:  {avg_accuracy:.3f} ± {np.std(fold_accuracies):.3f}")
    print(f"Average Precision: {avg_precision:.3f} ± {np.std(fold_precisions):.3f}")
    print(f"Average Recall:    {avg_recall:.3f} ± {np.std(fold_recalls):.3f}")

    print("\n=== Interpretation ===")
    if avg_accuracy > 0.52:
      print(f"✓ Model has edge! {avg_accuracy:.1%} accuracy beats random (50%)")
    else:
      print(f"✗ Model has NO edge. {avg_accuracy:.1%} accuracy is random guessing.")

    print("\nBaseline comparison:")
    print(f"  Random strategy: 50.0% accuracy")
    print(f"  Your ML model:   {avg_accuracy:.1%} accuracy")
    print(f"  Improvement:     {(avg_accuracy - 0.5)*100:.1f} percentage points")