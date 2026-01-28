quant_research_project/
├── data/                   # Git-ignored
│   ├── raw/                # Bronze (Original CSVs, SQLite)
│   ├── interim/            # Silver (Standardized/Cleaned)
│   └── processed/          # Gold (Modeling-ready panels)
├── src/                    # Source code as a package
│   ├── __init__.py
│   ├── data/               # Pipeline A: Loading & Preprocessing
│   │   ├── ingestion.py    # Bronze loading
│   │   ├── processor.py    # Silver transformations (rank-norm)
│   │   └── builder.py      # Gold joiner (t to t+1 alignment)
│   ├── models/             # Pipeline B & C
│   │   ├── baselines.py    # OLS, Lasso
│   │   ├── latent.py       # Autoencoders, K-Means
│   │   └── trainer.py      # Training loops & CV
│   └── evaluation/         # Metrics (R2_oos, Portfolio Sorts)
├── notebooks/              # Exploration & Visualization
├── tests/                  # Integrity checks (Leakage tests)
├── README.md
└── requirements.txt