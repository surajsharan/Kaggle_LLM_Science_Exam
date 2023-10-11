import pandas as pd
import numpy as np
from sklearn import model_selection
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def create_folds(data, num_splits):
    dfx = pd.get_dummies(data, columns=["answer"]).groupby(["id"], as_index=False).sum()
    cols = [c for c in dfx.columns if c.startswith("answer") or c == "id" and c != "answer"]
    dfx = dfx[cols]
    
    mskf = MultilabelStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    labels = [c for c in dfx.columns if c != "id"]
    dfx_labels = dfx[labels]
    dfx["kfold"] = -1
    
    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        dfx.loc[val_, "kfold"] = fold
    
    data = data.merge(dfx[["id", "kfold"]], on="id", how="left")
    
    return data




if __name__ == "__main__":

    train = pd.read_csv('inputs/train.csv')
    train = create_folds(train, num_splits=5)
    train.to_csv('inputs/train_folds.csv', index=False)
