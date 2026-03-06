# Self-Attention for Quantum Entanglement Prediction

[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jax](https://img.shields.io/badge/JAX-0.4.28-FFB612.svg?style=flat&logo=google)](https://github.com/google/jax)


The following repository is for a machine learning framework that predicts the Renyi-2 entropy of random pure-states using classical shadows via two models - self-attention and feed forward networks.

## Project Structure

```
project_root/
│── analytical_solution/
│   │── results/
│   └── analytical_results.py
|
│── models/
│   │── outputs/
│   │── main.py 
│   │── networks.py
│   │── testing.py
│   └── utils.py
│
│── notebooks/
│   │── analytics.ipynb
│   │── convergence_plots.ipynb 
│   └── results.ipynb
│
│── figures/
│
│── quantumSystem/
│   │── plot_bloch_sphere.py
│   │── generateData.py
│   │── pure_states.py
│   │── unitaries.py
│   └── utils.py
│
│── README.md
│── LICENSE
│── requirements.txt
└── .gitignore
```