# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pymde",
#     "numpy",
#     "pandas < 3.0",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymde
    from sklearn import preprocessing


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This notebook preprocesses the raw ACS dataset to produce the preprocessed data embedded in the 'Counties' notebook.
    """)
    return


@app.cell
def _():
    dataset = pymde.datasets.counties()
    return (dataset,)


@app.cell
def _(dataset):
    X = dataset.county_dataframe
    y = dataset.voting_dataframe
    return X, y


@app.cell
def _(X):
    X
    return


@app.cell
def _():
    totalpop_col_idx = 3
    col_idxes_to_normalize = [4, 5, 31]
    col_idxes_to_log = [totalpop_col_idx, 12, 13, 14, 15, 16]
    col_idxes_to_div_by_100 = [idx for idx in range(6, 11 + 1)]
    col_idxes_to_div_by_100 = col_idxes_to_div_by_100 + [idx for idx in range(17, 30)]
    col_idxes_to_div_by_100 = col_idxes_to_div_by_100 + [idx for idx in range(32, 36 + 1)]
    return (
        col_idxes_to_div_by_100,
        col_idxes_to_log,
        col_idxes_to_normalize,
        totalpop_col_idx,
    )


@app.cell
def _(
    X,
    col_idxes_to_div_by_100,
    col_idxes_to_log,
    col_idxes_to_normalize,
    totalpop_col_idx,
):
    col_names = X.columns
    for col_idx in col_idxes_to_normalize + col_idxes_to_log:
        col_idx_name = col_names[col_idx]
        X[col_idx_name] = X[col_idx_name].astype(float)
    for _row_idx, _row_obj in X.iterrows():
        totalpop_idx_name = col_names[totalpop_col_idx]
        totalpop = float(_row_obj[totalpop_idx_name])
        for col_idx in col_idxes_to_normalize:
            col_idx_name = col_names[col_idx]
            col_val = float(_row_obj[col_idx_name])
            X.at[_row_idx, col_idx_name] = col_val / totalpop
        for col_idx in col_idxes_to_log:
            col_idx_name = col_names[col_idx]
            col_val = float(_row_obj[col_idx_name])
            X.at[_row_idx, col_idx_name] = np.log(col_val)
        for col_idx in col_idxes_to_div_by_100:
            col_idx_name = col_names[col_idx]
            col_val = float(_row_obj[col_idx_name])
            X.at[_row_idx, col_idx_name] = col_val / 100.0
    return


@app.cell
def _(X):
    for _row_idx, _row_obj in X.iterrows():
        X.at[_row_idx, 'MeanCommute'] = X.at[_row_idx, 'MeanCommute'] / 60.0
    return


@app.cell
def _(y):
    y.rename(columns={"combined_fips":"CountyId"}, inplace=True)
    return


@app.cell
def _(X, y):
    Xy = X.merge(y, on="CountyId")
    return (Xy,)


@app.cell
def _(Xy):
    drop_cols = ["Unnamed: 0", "state_abbr", "county_name", "per_point_diff", "diff", "total_votes"]
    Xy.drop(columns=drop_cols, inplace=True)
    return


@app.cell
def _(Xy):
    y_many_cols = ['votes_dem', 'votes_gop', 'per_dem', 'per_gop', 'CountyId', 'State', 'County']
    y_many = Xy[y_many_cols]
    y_many = y_many.copy()
    X_1 = Xy.drop(columns=y_many_cols)
    return X_1, y_many


@app.cell
def _(X_1):
    X_2 = preprocessing.scale(X_1.to_numpy())
    return (X_2,)


@app.cell
def _(y_many):
    y_1 = pd.DataFrame(y_many['votes_gop'] <= y_many['votes_dem'])
    y_1 = y_1.to_numpy(dtype=np.int32)
    y_1 = y_1.flatten()
    return


@app.cell
def _(y_many):
    y_prop = pd.DataFrame(y_many["per_dem"]).to_numpy(dtype=np.float64)[:,0]
    return (y_prop,)


@app.cell
def _(X_2, y_prop):
    np.save('democratic_fraction.npy', y_prop)
    np.save('county_data.npy', X_2)
    return


if __name__ == "__main__":
    app.run()
