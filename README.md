conditional_asset_pricing/
│
├── README.md
├── requirements.txt
│
├── config/
│   ├── paths.yaml
│   ├── model_params.yaml
│   └── experiment_registry.yaml
│
├── data/
│   ├── raw/
│   │   ├── crsp_monthly.csv
│   │   └── characteristics_raw.csv
│   │
│   ├── processed/
│   │   ├── panel_standardized.parquet
│   │   └── panel_with_clusters.parquet
│   │
│   └── metadata/
│       ├── variable_descriptions.csv
│       └── sample_summary.json
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── clean_data.py
│   │   ├── standardize.py
│   │   └── split.py
│   │
│   ├── pricing_models/
│   │   ├── linear.py
│   │   ├── trees.py
│   │   ├── neural_nets.py
│   │   └── base.py
│   │
│   ├── latent_structure/
│   │   ├── __init__.py
│   │   ├── kmeans.py
│   │   ├── autoencoder.py
│   │   ├── som.py
│   │   ├── tda.py
│   │   └── validation.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── dm_test.py
│   │   ├── portfolio.py
│   │   └── diagnostics.py
│   │
│   └── utils/
│       ├── logging.py
│       ├── plotting.py
│       └── helpers.py
│
├── experiments/
│   ├── 01_baseline_linear.yaml
│   ├── 02_nonlinear_supervised.yaml
│   ├── 03_latent_kmeans.yaml
│   ├── 04_latent_autoencoder.yaml
│   └── 05_latent_tda.yaml
│
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_characteristic_space.ipynb
│   ├── 02_latent_structure_exploration.ipynb
│   ├── 03_pricing_comparison.ipynb
│   └── 04_economic_interpretation.ipynb
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── logs/
│
└── papers/
    └── research_memo.pdf
