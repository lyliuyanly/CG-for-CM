
# Latent-Class Choice Model with Different Gradient Estimation

## Overview

This Python script implements a **latent class choice model** for route selection behavior based on the **Swissmetro route choice dataset**. It allows for flexible gradient computation using:

- **Automatic differentiation**
- **Analytical gradient formulation**
- **Numerical finite difference**

The model accounts for individual-level panel data and performs **maximum likelihood estimation** using the BFGS optimization algorithm from TensorFlow Probability.

## File: `Mixed_Swissdata_test.py`

### Core Features

- **Two latent classes (LC1 and LC2)** with separate utility parameters.
- Uses panel-style choice data (9 observations per individual).
- Implements full **log-likelihood computation** and estimation.
- Supports analytical/numerical gradient derivation for gradient checking or custom optimization.
- Outputs estimation results including:
  - Coefficients
  - Standard errors (classical & robust)
  - T-ratios
  - AIC, BIC, LAMADA
  - Gradients and correlation matrices

## Data Requirements

- Input file: `apollo_swissRouteChoiceData.csv`
- Format:
  - Contains attributes: `tt1, tc1, hw1, ch1, tt2, tc2, hw2, ch2`
  - Choice: `choice` (1 or 2)
  - Availability: `av1, av2`

Place this CSV in the same directory or set `inputLocation` accordingly.

## How to Run

```bash
python Mixed_Swissdata_test.py
```

## Main Components

| Section | Description |
|--------|-------------|
| `model_fun()` | Computes latent-class weighted choice probabilities |
| `model_fun_analytical_mean()` | Computes analytical gradient mean (optional) |
| `cost_fun()` | Negative log-likelihood function |
| `numerical_gradient()` | Finite difference gradient for validation |
| `loss_gradient()` | Wrapper for automatic/analytical/numerical gradients |
| `tfp.optimizer.bfgs_minimize` | Main estimation using BFGS |

## Output

- Estimation results are saved as:
  ```
  output_params_testout_LC2-ND.csv
  ```
- Console prints:
  - LL(initial), LL(final), LAMADA
  - AIC, BIC, time
  - Gradient checks (optional)

## Dependencies

```bash
pip install tensorflow tensorflow-probability pandas numpy matplotlib scipy scikit-learn
```

## Notes

- Ensure the dataset size is a multiple of `number_choices = 9` (i.e., panel data per individual).
- You can change the gradient type (`automatic`, `analytical`, `numerical`) via the variable `gradient_type`.
