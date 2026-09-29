# Math

This file walks through the loss function, the gradient, and the update rule behind every model in this repository. The notation matches the code: `X` is the feature matrix of shape `(m, n)`, `w` is the weight vector, `b` is the bias, `y` is the target, and `m` is the number of training samples.

## Linear Regression

The model predicts

$$\hat{y} = Xw + b$$

and is trained to minimize the mean squared error

$$L(w, b) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

Differentiating with respect to `w` and `b` gives

$$\frac{\partial L}{\partial w} = -\frac{2}{m} X^T (y - \hat{y})$$

$$\frac{\partial L}{\partial b} = -\frac{2}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)$$

Each training step moves `w` and `b` a small distance against the gradient, scaled by the learning rate `lr`:

$$w \leftarrow w - \text{lr} \cdot \frac{\partial L}{\partial w}, \qquad b \leftarrow b - \text{lr} \cdot \frac{\partial L}{\partial b}$$

This is exactly what `LinearRegressionModel.fit` computes each epoch.

## Ridge Regression (L2 penalty)

Ridge adds a penalty proportional to the squared size of the weights, so large weights cost more:

$$L(w, b) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{n} w_j^2$$

The penalty term's derivative with respect to `w` is `2 * alpha * w`, so the full gradient is the plain linear regression gradient plus this extra term:

$$\frac{\partial L}{\partial w} = -\frac{2}{m} X^T (y - \hat{y}) + 2\alpha w$$

The bias is not penalized, so `db` is unchanged from plain linear regression. Because the penalty grows with `w`, larger weights are pulled back toward zero more strongly than small ones, which is why Ridge shrinks every weight without forcing any of them to exactly zero.

## Lasso Regression (L1 penalty)

Lasso penalizes the absolute size of the weights instead of the squared size:

$$L(w, b) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{n} |w_j|$$

The absolute value function is not differentiable at zero, so the code uses its subgradient, `sign(w)`, which is `+1` for positive weights, `-1` for negative weights, and `0` at exactly zero:

$$\frac{\partial L}{\partial w} = -\frac{2}{m} X^T (y - \hat{y}) + \alpha \cdot \text{sign}(w)$$

Unlike Ridge, this penalty pulls every nonzero weight toward zero by the same fixed amount `alpha` on every step, regardless of how large the weight is. A small weight can get pushed past zero and stay there, which is why Lasso tends to produce sparse solutions, some weights become exactly zero, while Ridge only shrinks them.

## Binary Logistic Regression

Binary classification needs predictions between 0 and 1, so the linear score `z = Xw + b` is passed through the sigmoid function:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

`p` is interpreted as the predicted probability of the positive class. The model is trained to minimize binary cross entropy, also called log loss:

$$L(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

The sigmoid and the log loss are a convenient pair: when the gradient of `L` with respect to `z` is worked out, the derivative of the sigmoid cancels out almost entirely, leaving

$$\frac{\partial L}{\partial w} = \frac{1}{m} X^T (p - y), \qquad \frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (p_i - y_i)$$

This has the same shape as the linear regression gradient, with `p` standing in for the prediction, which is why the two `fit` methods look so similar in code despite having different loss functions.

## Multinomial Logistic Regression

With more than two classes, each class gets its own column of weights, so `w` becomes a matrix of shape `(n, k)` and `b` becomes a vector of length `k`, where `k` is the number of classes. The labels are one hot encoded into `Y`, a `(m, k)` matrix where each row has a single 1 in the column of the true class.

The linear score for every class is

$$Z = XW + b$$

and softmax turns the `k` scores for each sample into a probability distribution over the classes:

$$P_{i,c} = \frac{e^{Z_{i,c}}}{\sum_{j=1}^{k} e^{Z_{i,j}}}$$

`softmax` in `metrics.py` subtracts the row maximum from `Z` before exponentiating. This does not change the result, since it cancels out in the ratio, but it keeps the exponential from overflowing for large scores.

Training minimizes categorical cross entropy:

$$L(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{k} Y_{i,c} \log(P_{i,c})$$

As with the binary case, softmax paired with categorical cross entropy gives a clean gradient:

$$\frac{\partial L}{\partial W} = \frac{1}{m} X^T (P - Y), \qquad \frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (P_i - Y_i)$$

which again has the same `prediction minus target` shape as the other two models, now averaged over samples and computed per class.

## Why gradient descent works here

Every loss function above is convex in `w` and `b` on its own (Lasso's penalty is convex but not smooth at zero, which is why it uses a subgradient rather than a true gradient there). A convex loss has no local minima to get stuck in, only a single global minimum, so repeatedly stepping against the gradient is guaranteed to move toward that minimum as long as the learning rate is small enough not to overshoot it. `fit` also stops early once the loss stops improving by more than `tol` between epochs, since further steps at that point are not changing the result enough to justify the extra computation.

## Matching scikit-learn's alpha

scikit-learn's `Ridge` and `Lasso` scale their penalty terms differently from this repository. `main.py` converts `alpha` before creating the scikit-learn models so that both sides are minimizing the same objective:

scikit-learn's `Ridge` minimizes

$$\sum_{i=1}^{m} (y_i - \hat{y}_i)^2 + \alpha_{sk} \sum_{j=1}^{n} w_j^2$$

which is `m` times this repository's Ridge loss with the mean squared error term expanded back into a sum, so `alpha_sk = alpha * m` makes the two penalties match.

scikit-learn's `Lasso` minimizes

$$\frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 + \alpha_{sk} \sum_{j=1}^{n} |w_j|$$

which uses `1 / (2m)` in front of the squared error instead of `1 / m`, so matching the L1 term requires `alpha_sk = alpha / 2`.