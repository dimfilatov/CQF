# Module 4: Machine Learning

This folder contains examples of optimization, regression, classification,
feature selection, tree models, and boosting. Several scripts expect datasets
under `./data` relative to the current working directory; market-data examples
may also require the optional `quantmod` package or external data access.

## Python module overview

| Module | Summary |
| --- | --- |
| `feature_selection.py` | Demonstrates regression feature selection using variance inflation factor (VIF), `SelectKBest` with an F-test, recursive feature elimination (RFE), and cross-validated RFE (RFECV). It fits a linear-regression pipeline to the selected features and reports its in-sample R-squared. |
| `gradient_boosting.py` | Builds an XGBoost binary classifier for next-period SPY price direction. It engineers rolling return and volatility features, uses time-ordered train/test and cross-validation splits, searches hyperparameters, and reports classification metrics and plots. See the detailed walkthrough below. |
| `gradient_descent.py` | Generates synthetic data from a linear relationship with Gaussian noise, estimates the coefficients using batch gradient descent on mean squared error, and plots the loss history. |
| `grid_search.py` | Creates a standardized logistic-regression pipeline and exhaustively tests candidate regularization penalties and `C` values with `GridSearchCV`. |
| `knn.py` | Generates a two-class synthetic blob dataset, prints the nearest observations to a selected point using Euclidean distance, and plots an sklearn K-nearest-neighbors decision boundary. Its `__main__` block also calls `load_data` and `run_pipeline`, which are not implemented on the class. |
| `linear_regression.py` | Fetches market data, creates price/return/technical-indicator features, removes highly correlated features, then compares Linear Regression, Lasso, Ridge, and ElasticNet on a chronological holdout using R-squared, MSE, and RMSE. Requires `quantmod`. |
| `loanclub.py` | Separates loan features and labels, calculates entropy and information gain for a selected feature, and trains/plots a decision-tree classifier on supplied LendingClub train/test/validation data. |
| `logistic_regression.py` | Fetches market data and creates technical indicators for binary trend classification. It removes highly correlated features, scales selected columns robustly and the others with standard scaling, fits class-balanced logistic regression, evaluates it, and calculates a simple test-period trading signal and return summary. Requires `quantmod`. |
| `outlier_transform.py` | Implements a scikit-learn-style transformer that clips columns to configured lower/upper percentile bounds. Its example applies this to several SPY return horizons and plots the original and clipped distributions. |
| `quadratic_loss_boosting.py` | Demonstrates regression boosting on synthetic quadratic data. It fits shallow regression trees sequentially, with each later tree fitted to the residuals from the current ensemble, and plots each stage. See the detailed walkthrough below. |

## `quadratic_loss_boosting.py`: residual-based regression boosting

This module is a small, explicit demonstration of additive boosting for
regression. It generates one predictor `X` and a response
`y = 3 * X^2 + error`, where `error` is Gaussian noise. The generated data are
stored on the `QuadraticLossBoosting` instance.

### Estimation during `fit()`

The constructor parameter `n_estimators` is the number of trees in the
ensemble. Each tree is a `DecisionTreeRegressor` with a fixed `max_depth=2`.
There is no hyperparameter search, sample weighting, separate validation set,
or learning-rate/shrinkage parameter in this implementation.

1. The first depth-2 tree is fitted directly to `(X, y)`. This is equivalent to
   starting with a zero prediction and fitting the initial residual `y - 0`.
2. For each subsequent stage `m`, the existing trees' predictions are summed
   on the training `X`. The residual is calculated for each observation:
   `residual_m = y - current_prediction`.
3. A new depth-2 regression tree is fitted from `X` to `residual_m`. Its output
   approximates the remaining error as a function of `X`.
4. The tree is appended to the ensemble. The fitting code plots the updated
   ensemble, its newest tree contribution, and the residuals before and after
   that contribution.

Why do residuals represent the negative gradient? Let `F(x)` be the current
prediction for an observation with target `y`. Use half the squared error as
the loss:

`L(y, F) = 1/2 * (y - F)^2`

Differentiating with respect to the current prediction gives:

`dL/dF = F - y`

The negative gradient is therefore:

`-dL/dF = y - F = residual`

So, at each later stage, fitting a regression tree to `y - F(x)` is fitting
the direction in prediction space that most reduces squared loss locally.
For a leaf whose observations have residuals `r_i`, the constant prediction
that minimizes their squared errors is their mean residual,
`leaf_value = mean(r_i)`. The tree partitions feature space into leaves so
that adding these leaf values reduces the overall residual error.

For example, if an observation has `y = 10` and the current ensemble predicts
`F(x) = 7`, its residual is `3`; the next tree is trained to add about `3`
for observations in a similar region. If the current prediction is `12`, its
residual is `-2`, so the next tree should pull that region's prediction down.

The model's prediction after `m` trees is the sum of the individual tree
predictions:

`F_m(x) = T_1(x) + T_2(x) + ... + T_m(x)`

The implementation does not multiply each contribution by a learning rate,
so each newly fitted tree is added at full strength.

### Making predictions

`predict(X)` asks every fitted tree for a numeric prediction on the supplied
rows and sums those values. The returned values are regression estimates in
the same units as `y`; they are not class labels or probabilities. In the
example data, the intended signal is the conditional mean near `3 * X^2`,
while individual observed `y` values also contain noise.

The script's example uses `n_estimators=3`, so predictions are the sum of
three depth-2 tree outputs. The data generation is random and does not set a
seed, so a new run can produce a different sample and fitted model.

## `gradient_boosting.py`: XGBoost trend classification

Despite the filename, this module uses XGBoost's `XGBClassifier`, rather than
scikit-learn's `GradientBoostingClassifier`. The intended positive class is
label `1`, meaning the next adjusted close is greater than 99.5% of the
current adjusted close.

### Features, labels, and chronological holdout

`create_features()` calculates log returns from `Adj Close`, then rolling
return sums and rolling standard deviations for windows 10, 15, ..., 60. It
creates the next-period target as:

`Label = 1 if next Adj Close > 0.995 * current Adj Close, else 0`

Rows with unavailable rolling values or a missing next close are dropped.
The feature matrix excludes raw OHLC prices, adjusted close, the one-period
return, and the label; the rolling return and volatility columns remain.
`split_data()` takes the first `1 - test_size` portion for training and the
later observations for testing, without shuffling. The test portion is held
out of model selection and is used for final evaluation.

### How the classifier estimates its model

An XGBoost classifier constructs trees sequentially. Internally, its binary
logistic objective evaluates the current model's prediction errors using
gradients and Hessians, and new tree splits are selected to improve the
objective while accounting for regularization. Gradients and Hessians do not
directly report a row's classification error. Instead, they describe how
the loss changes as the model's raw score changes, and how curved that loss
is locally; XGBoost uses them to choose useful splits and leaf values.

For binary logistic classification, let `z_i` be the model's raw score
(margin) for row `i`, and convert it to a probability with the sigmoid:

`p_i = sigmoid(z_i) = 1 / (1 + exp(-z_i))`

For label `y_i` in `{0, 1}`, the binary logloss for one row is:

`l_i = -[y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]`

If the row has sample weight `w_i`, its contribution is `w_i * l_i`.
Differentiating this loss with respect to the raw score gives the first and
second derivatives used by XGBoost:

`g_i = w_i * (p_i - y_i)`

`h_i = w_i * p_i * (1 - p_i)`

The gradient `g_i` gives the local direction of loss change. If `y_i=1` and
`p_i=0.8`, then `g_i=-0.2*w_i`: increasing the raw score (and thus the
probability) locally lowers the loss. If `y_i=0` and `p_i=0.8`, then
`g_i=0.8*w_i`: the score should move down. The Hessian `h_i` is the local
curvature; for logistic loss it is largest near `p_i=0.5` and approaches
zero as `p_i` approaches 0 or 1. The pair `(g_i, h_i)` lets XGBoost form a
second-order approximation to the effect of adding a tree, rather than
trying every possible change in prediction directly.

At boosting round `t`, a new tree contributes `f_t(x_i)` to the current
margin. A second-order Taylor expansion of the loss around the current margin
approximates the change in training objective as:

`sum_i [g_i * f_t(x_i) + 1/2 * h_i * f_t(x_i)^2] + Omega(f_t)`

Here `Omega(f_t)` penalizes tree complexity. In the simplified L2-only case,
`Omega` includes a penalty `1/2 * lambda * sum_j (w_j)^2`, where `w_j` is
the score assigned to leaf `j`; XGBoost also uses split penalties such as
`gamma`. This approximation explains how derivatives connect predictions
to tree fitting: a candidate tree is scored by how much it is expected to
lower the objective locally, after its complexity penalty.

For a candidate leaf containing rows `I`, aggregate the derivatives:

`G = sum(g_i for i in I)` and `H = sum(h_i for i in I)`

With L2 leaf regularization `lambda` and no L1 penalty, the optimal constant
leaf score follows by minimizing the leaf's approximate objective
`G * w + 1/2 * (H + lambda) * w^2` with respect to `w`:

`leaf_score = -G / (H + lambda)`

Substituting this value back gives the leaf's improvement score (ignoring
terms common to both tree structures):

`score(I) = 1/2 * G^2 / (H + lambda)`

For a proposed split into left and right children, XGBoost's gain is:

`Gain = 1/2 * [G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - G^2/(H + lambda)] - gamma`

Here `G` and `H` are for the parent and `G_L`, `H_L`, `G_R`, and `H_R` are
for its children. A split is useful when this gain is positive and its
children also meet `min_child_weight`. Thus the derivatives turn the loss
into a numerical criterion for both the amount a leaf should adjust scores
and whether a split is worth adding.

The classifier's raw tree scores (margins) are converted to class
probabilities; the probability for class `1` is the model's estimated
positive-class probability. A class prediction is then made from the
probability using the classifier's decision threshold.

Before fitting, the module calculates balanced sample weights from the
training labels. These increase the training contribution of observations
from the less frequent class. They affect fitting but do not change the
meaning of predicted probabilities or the ROC-AUC metric.

`fit()` calls `tune_model()`, which creates a `RandomizedSearchCV` with:

- ROC-AUC as the selection score.
- `TimeSeriesSplit(n_splits=5, gap=1)` so folds preserve order and each fold
  leaves a one-row gap between its training and validation sections.
- `n_iter=5`, so only five sampled parameter combinations are evaluated.
- `random_state=42` for repeatable parameter sampling.
- `refit=True`, so after selection the winning parameter combination is
  fitted again on all of the outer training data.

There are two different optimization/evaluation roles here:

1. **Within each XGBoost fit, the training objective is regularized binary
   logistic loss.** `XGBClassifier` uses the binary logistic objective by
   default. Its tree-building algorithm uses the logloss derivatives above
   to find score updates and splits, while penalizing model complexity. The
   model is trained to improve this objective; it does not train by directly
   maximizing ROC-AUC.
2. **Across candidate parameter combinations, the search score is ROC-AUC.**
   `RandomizedSearchCV(scoring="roc_auc")` computes ROC-AUC on each
   chronological validation fold and chooses the combination with the
   highest mean fold score. ROC-AUC measures ranking quality across
   thresholds, rather than the quality of one fixed threshold. It is a
   selection metric here, not the differentiable loss used to build trees.

The explicit `eval_metric="logloss"` passed to `_make_model()` is XGBoost's
evaluation metric for any supplied evaluation set. This implementation
doesn't pass an `eval_set` to `.fit()`, so it does not drive early stopping
or parameter selection in the current workflow. The CV selection metric is
separately set by `RandomizedSearchCV` to ROC-AUC.

The dictionary `SEARCH_PARAM_DISTRIBUTIONS` provides the candidate values for
the randomized search:

| Parameter | Candidate values | Meaning and typical impact |
| --- | --- | --- |
| `learning_rate` | `0.05`, `0.10`, `0.15`, `0.20`, `0.25`, `0.30` | Shrinks each boosting step. Lower values make updates more gradual and commonly need more trees; higher values learn more quickly but can overfit or skip over a useful solution. |
| `max_depth` | `3`, `4`, `5`, `6`, `8`, `10`, `12`, `15` | Maximum depth of each tree. Larger values allow more complex rules but can fit noise and increase computation. |
| `min_child_weight` | `1`, `3`, `5`, `7` | Minimum sum of Hessian weights in a child node for a split. Larger values discourage splits supported by only a small or weakly informative group, making the trees more conservative. It is not simply a minimum row count. |
| `gamma` | `0.0`, `0.1`, `0.2`, `0.3`, `0.4` | Minimum loss reduction needed to add a split. Larger values require stronger improvement before a tree can grow another branch. |
| `colsample_bytree` | `0.3`, `0.4`, `0.5`, `0.7` | Fraction of input features sampled for each tree. Lower values add randomness and may reduce overfitting; too little feature availability can omit useful predictors. |
| `n_estimators` | `100`, `200`, `300` | Number of sequential trees/boosting rounds. More trees provide more opportunities to improve the fit, but take longer and may overfit; the best count interacts with `learning_rate`. |

The six lists represent `6 * 8 * 4 * 5 * 4 * 3 = 11,520` possible
combinations. Five are sampled and scored, not all 11,520. Thus, the selected
combination is best among the five tried, not guaranteed to be globally best.
The search optimizes ROC-AUC, which measures how well the scores rank positive
examples above negative examples across thresholds; it does not select a
classification threshold.

The search only selects the parameters listed in
`SEARCH_PARAM_DISTRIBUTIONS`, based on mean CV ROC-AUC. In that dictionary,
`gamma` and `min_child_weight` regularize tree growth by rejecting weak or
insufficiently supported splits; `max_depth`, `colsample_bytree`, and
`n_estimators` also constrain model complexity. Other regularization
parameters are not searched. For example, XGBoost's `reg_lambda` (the `lambda`
in the leaf formula above) stays at its library default unless explicitly
overridden, as does `reg_alpha` (L1 regularization). Therefore this search
does not find a universal or joint optimum for every XGBoost regularizer: it
finds the best of five sampled combinations under the stated CV score, with
the unlisted settings held fixed.

Other settings fixed by `_make_model()` are `verbosity=0`,
`eval_metric="logloss"`, `random_state=42`, and `n_jobs=1`. The search itself
uses `n_jobs=-1` to parallelize candidate/fold fits.

### From fitted scores to predictions

After the search, `self.model` is assigned `self.search.best_estimator_`.
Because `refit=True`, this is already fitted on all of `X_train` using the
selected parameters. `predict_proba()` returns probabilities in the order
listed by `model.classes_`; the code locates class `1` and uses that
probability for ROC-AUC. `predict()` returns class labels using the
classifier's default binary decision threshold (normally 0.5).

`evaluate()` reports training and test accuracy and balanced accuracy, plus
test ROC-AUC, F1, precision, recall, and a classification report. The plotting
methods show the held-out test confusion matrix, ROC curve, precision-recall
curve, and feature importance (gain by default). The precision-recall curve
and ROC curve examine model scores across thresholds; they are not restricted
to the default `predict()` threshold.

The full pipeline result also includes `best_params`, the mean best CV ROC-AUC
(`best_cv_roc_auc`), and the standard deviation across that winning
configuration's folds (`best_cv_roc_auc_std`).

### Cross-validation sample-weight note

Balanced weights are calculated once from all `y_train` and passed to
`RandomizedSearchCV`; scikit-learn slices those weights for each fold's fit.
Consequently, each fold's training weights are based on class frequencies
from the whole outer training period, including its validation segment. For
strictly fold-local class weighting, calculate weights from each CV training
fold separately (for example, with a custom CV loop or estimator that
computes weights inside `fit`). The final refit uses weights computed from
the full outer training set, as intended.
