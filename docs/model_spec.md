# Model Architectures & Learning Dynamics

This document details the machine learning and deep learning architectures, mathematical formulation, loss functions, optimization techniques, and evaluation protocols for energy consumption forecasting.

---

## 1. Supported Model Families

To provide a robust progression from standard baselines to state-of-the-art models, the system architecture accommodates three core modeling paradigms:

```
                                ┌── Seasonal Naive / Baseline
                                │
    Forecasting Architectures ──┼── Gradient Boosted Trees (LightGBM / XGBoost)
                                │
                                └── Deep Sequence Models (LSTM / GRU / TCN / Transformer)
```

1. **Statistical & Heuristic Baselines**:
   - Seasonal Persistence: $\hat{y}_{t+h} = y_{t+h-S}$ where $S \in \{24, 168\}$.
   - Serves as the zero-cost benchmark that all learning models must surpass.

2. **Gradient Boosted Decision Trees (GBDT - LightGBM/XGBoost)**:
   - High sample efficiency, handles tabularized lag features and calendar features effectively.
   - Fits additive ensembles of decision trees minimizing empirical risk via gradient descent in function space.

3. **Recurrent & Deep Sequential Models (LSTM / GRU / Temporal CNN)**:
   - Capable of learning non-linear temporal dynamics and long-range dependencies directly from sequential tensors:
     $$X \in \mathbb{R}^{B \times T_{\text{in}} \times D} \longrightarrow \hat{Y} \in \mathbb{R}^{B \times T_{\text{out}}}$$
     where $B$ is batch size, $T_{\text{in}}$ is lookback window length (e.g. 168 hours / 1 week), $D$ is feature dimensionality, and $T_{\text{out}}$ is forecast horizon (e.g. 24 hours).

---

## 2. Mathematical Formulations & Learning Mechanics

### Recurrent Cell Computation (LSTM)
For each time step $t$ within the lookback window:
$$\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad &&\text{(Forget Gate)} \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad &&\text{(Input Gate)} \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \quad &&\text{(Candidate Memory)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad &&\text{(Cell State Update)} \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad &&\text{(Output Gate)} \\
h_t &= o_t \odot \tanh(c_t) \quad &&\text{(Hidden State Output)}
\end{aligned}$$

The final hidden state $h_{T_{\text{in}}}$ is mapped through dense projection layers with dropout regularizers to output the forecast vector $\hat{Y} = [\hat{y}_{t+1}, \hat{y}_{t+2}, \dots, \hat{y}_{t+T_{\text{out}}}]^T$.

---

## 3. Loss Functions & Optimization

### Point Forecasting Losses
- **Mean Squared Error (MSE)**: Penalizes larger peak prediction errors strongly:
  $$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
- **Huber Loss**: Smooth transition between $\mathcal{L}_2$ (for small errors) and $\mathcal{L}_1$ (for large outlier spikes):
  $$\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases}
  \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\
  \delta \left(|y - \hat{y}| - \frac{1}{2}\delta\right) & \text{otherwise}
  \end{cases}$$

### Probabilistic Uncertainty Loss (Quantile / Pinball Loss)
For predicting lower (10th), median (50th), and upper (90th) confidence bounds:
$$\mathcal{L}_{q}(y, \hat{y}) = \max(q(y - \hat{y}), (q - 1)(y - \hat{y}))$$

### Optimization & Regularization Strategy
- **Optimizer**: AdamW with decoupled weight decay ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$).
- **Learning Rate Scheduler**: Cosine Annealing with Warmup.
- **Gradient Clipping**: Maximum $L_2$ norm threshold of $1.0$ to prevent exploding gradients in recurrent architectures.
- **Early Stopping**: Monitored on chronological validation loss with a patience window (typically 10–15 epochs).

---

## 4. Evaluation Metrics Benchmark

All models are systematically benchmarked using standard time-series regression metrics:

1. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$
3. **Mean Absolute Percentage Error (MAPE)**:
   $$\text{MAPE} = \frac{100\%}{N}\sum_{i=1}^N \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
4. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

