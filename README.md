 California Housing — Data Quality Audit & Exploratory Analysis

A reproducible Python project that audits the quality of the California Housing dataset and explores its numerical distributions, correlations, and selected feature/target relationships. This is a descriptive analysis project, not a predictive model.

 Project highlights

- Checks the dataset schema, missingness, exact duplicate rows, non-finite values, basic value ranges, and unusual observations.
- Produces descriptive statistics, Pearson and Spearman correlations, and a comparison of target summaries across median-income quintiles.
- Generates six visualizations and machine-readable summary tables.
- Preserves the input CSV: the analysis does not impute values, delete rows, round target values, or create engineered features.
- Includes external unit tests for the schema contract and range-check behavior.

 Dataset and provenance

The California Housing dataset is described in the official scikit-learn documentation. It contains 20,640 census block-group observations and eight numeric input features plus the `MedHouseVal` target. The observations derive from the 1990 U.S. Census; rows are geographic block-group summaries, not individual house listings. The target is expressed in units of $100,000, and the documentation describes an upper cap of 5.0.

- [scikit-learn dataset loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [scikit-learn dataset description](https://scikit-learn.org/stable/datasets/real_world.html)
- [Original StatLib dataset archive](https://lib.stat.cmu.edu/datasets/houses.zip)

The raw CSV is not included in this repository. This avoids redistributing the dataset without separately assessing its applicable terms. Use `prepare_dataset.py` to obtain the dataset via scikit-learn's documented loader and save the expected local CSV. Dataset terms remain separate from the license for this project’s code.

 Key findings in the published analysis snapshot

The reviewed report and tables describe the supplied CSV artifact used for this analysis:

- 20,640 rows × 9 columns.
- 0 missing cells, 0 exact duplicate rows, and 0 non-finite numeric values.
- Pearson correlation between `MedInc` and `MedHouseVal`: 0.6881; Spearman correlation: 0.6768.
- The CSV’s observed maximum `MedHouseVal` is 5.00001; 992 rows are at least 5.0 and 965 are exactly 5.00001.
- Mean target rises from 1.1840 in the lowest median-income quintile to 3.3499 in the highest quintile, in the dataset’s units of $100,000.

These are descriptive associations, not causal findings or current-market price estimates. The report documents the exact target-value discrepancy rather than altering the data. Re-created data and outputs should be checked when software versions or source artifacts differ.

 Repository structure

```text
california-housing-data-quality-eda/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── housing_analysis.py        Preserved byte-for-byte from the Run 2 script
├── prepare_dataset.py         Fetches the documented dataset and writes data/california_housing.csv
├── data/
│   └── README.md               Data setup instructions; raw CSV is excluded
├── tests/
│   └── test_housing_analysis.py
└── reports/
    ├── housing_analysis_report.md
    ├── figures/                Six report figures
    └── tables/                 Aggregated analysis tables
```

The script writes execution outputs under `outputs/` by default. This generated directory is excluded from Git because runtime metadata contains machine-specific execution paths. The reviewed report snapshot and selected tables/figures are stored under `reports/` instead.

 Setup and execution (Windows PowerShell)

Use Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python prepare_dataset.py
python -m pip check
python -m unittest discover -s tests -v
python housing_analysis.py
```

`prepare_dataset.py` may need an internet connection on the first run to retrieve the dataset through scikit-learn. It validates the expected shape and column order before writing the CSV. The analysis script reads `data/california_housing.csv` by default and writes generated files to `outputs/`.

To use a different CSV or output directory:

```powershell
python housing_analysis.py --input path\to\california_housing.csv --output-dir path\to\outputs
```

 Interpretation and limitations

- Historical 1990 Census context: these are not current California housing listings or current-market estimates.
- Each row represents a census block group, not an individual property.
- The target is in units of $100,000, and high-end values are affected by the documented cap. The analyzed CSV includes values of 5.00001; the report preserves that observation.
- Large `AveRooms`, `AveBedrms`, and `AveOccup` values can reflect aggregate ratios and should be investigated, not automatically deleted.
- Correlation does not establish causation; geography, aggregation, and unmeasured variables may affect relationships.
- No predictive model is trained or tuned.

 Tests and quality notes

The `tests/` suite independently exercises the declared schema and verifies that the range-check function flags deliberately invalid sample values. Once the dataset is prepared, a dataset-contract test checks its actual dimensions, columns, missingness, and finite numeric values. The production script's existing internal assertions are preserved unchanged for provenance; the external tests supplement them.

 Acknowledgements and AI assistance

Dataset provenance and interpretation notes are credited to the official scikit-learn documentation and the original StatLib source. AI assistance was used during project analysis, code/documentation review, and portfolio preparation; the repository should not be represented as exclusively manual work.

 License

The included MIT license applies to the project code and original project materials where the author has rights to license them. It does not relicense the underlying dataset or third-party content. Review the dataset source terms separately before redistributing raw data.
