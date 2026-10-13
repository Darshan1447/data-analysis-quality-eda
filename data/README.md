# Dataset setup

The raw `california_housing.csv` is intentionally not committed to this repository. From the repository root, create it with:

```powershell
python -m pip install -r requirements.txt
python prepare_dataset.py
```

The helper calls `sklearn.datasets.fetch_california_housing(as_frame=True)` and saves the returned frame, including `MedHouseVal`, to `data/california_housing.csv`. An internet connection may be required the first time the loader runs.

Source notes: [scikit-learn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), [dataset description](https://scikit-learn.org/stable/datasets/real_world.html), and [original StatLib archive](https://lib.stat.cmu.edu/datasets/houses.zip).

The repository does not make a claim about redistribution rights for the raw dataset. The code license and dataset terms are separate.
