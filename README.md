quant_research_project/ <br>
├── data/                   # Git-ignored <br>
│   ├── raw/                # Bronze (Original CSVs, SQLite) <br>
│   ├── interim/            # Silver (Standardized/Cleaned) <br>
│   └── processed/          # Gold (Modeling-ready panels) <br>
├── src/                    # Source code as a package <br>
│   ├── __init__.py <br>
│   ├── data/               # Pipeline A: Loading & Preprocessing <br>
│   │   ├── ingestion.py    # Bronze loading <br>
│   │   ├── processor.py    # Silver transformations (rank-norm) <br>
│   │   └── builder.py      # Gold joiner (t to t+1 alignment) <br>
│   ├── models/             # Pipeline B & C <br>
│   │   ├── baselines.py    # OLS, Lasso <br>
│   │   ├── latent.py       # Autoencoders, K-Means <br>
│   │   └── trainer.py      # Training loops & CV <br>
│   └── evaluation/         # Metrics (R2_oos, Portfolio Sorts) <br>
├── notebooks/              # Exploration & Visualization <br>
├── tests/                  # Integrity checks (Leakage tests) <br>
├── README.md <br>
└── requirements.txt <br>
