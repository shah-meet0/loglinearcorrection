## Package structure

loglinearcorrection/
├── __init__.py                    # Package initialization
├── model.py                        # Main estimation model
│   └── DoublyRobustElasticityEstimatorModel (DREEM)
├── nonparametric.py               # Nonparametric regression models
│   ├── NPModel                    # Base nonparametric model
│   ├── NPModelResults             # Results container with predictions
│   ├── NNModel                    # Neural network implementation
│   ├── NNModelResults             # NN-specific results
│   ├── NNModelScore               # Score matching variant
│   └── NNModelScoreResults        # Score matching results
├── neural_network_models.py       # PyTorch network architectures
│   ├── FeedForwardNNModel         # MLP implementation
│   └── SlicedScoreMatchingLoss    # Score matching loss
├── utils.py                        # Utility functions
│   ├── _apply_fixed_effects()     # Fixed effects demeaning
│   └── _detect_variable_types()   # Binary/continuous detection
├── density.py (TODO)              # Density estimation
│   ├── DensityModel               # Density estimator
│   └── DensityModelResults        # Density predictions
└── results.py (TODO)              # Results classes
    └── DREEMR                     # Main results container

┌─────────────────────────────────────────────────────────┐
│                    DREEM.fit()                          │
│                  (Main Estimator)                       │
└────────┬────────────────────────┬───────────────────────┘
         │                        │
         ▼                        ▼
┌────────────────────┐   ┌────────────────────┐
│     NPModel        │   │   DensityModel     │
│   (m(x) = E[e^u|x])│   │   (f(x) density)   │
└────────┬───────────┘   └────────┬───────────┘
         │                        │
         ▼                        ▼
┌────────────────────┐   ┌────────────────────┐
│  NPModelResults    │   │DensityModelResults │
│ - predict()        │   │ - predict()        │
│ - derivative()     │   │ - predict_semi_    │
│ - predict_semi_    │   │   elasticity()     │
│   elasticity()     │   └────────────────────┘
└────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                      DREEMR                             │
│  - elasticities: Dict[str, Dict]                        │
│  - beta: averaged OLS coefficients                      │
│  - fold_diagnostics:                                    │
│    - m_results: List[NPModelResults]                    │
│    - f_results: List[DensityModelResults]               │
│    - alpha_x: List[ndarray] (influence functions)       │
│    - theta_x: List[ndarray] (elasticity estimates)      │
└─────────────────────────────────────────────────────────┘


