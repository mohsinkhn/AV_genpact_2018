import numpy as np
import pandas as pd
import config
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_true, y_preds))

def main():
    print("Weighted averaging models")
    val_lgb1 = np.load(str(config.SAVE_PATH / "val_preds_lgbv4.npy"))
    val_etr1 = np.load(str(config.SAVE_PATH / "val_preds_etrv3.npy"))
    val_etr2 = np.load(str(config.SAVE_PATH / "val_preds_etrv4.npy"))
    val_nn1 = np.load(str(config.SAVE_PATH / "val_preds_nnv3.npy"))

    val2 = pd.read_csv(str(config.SAVE_PATH / "val2f2.csv"))
    y_val = np.log1p(val2["num_orders"])

    val_preds = 0.5 * val_lgb1 + 0.1 * val_etr1 + 0.1 * val_etr2 + 0.30 * val_nn1.flatten()
    print("Final validation RMSE is ", rmse(y_val, val_preds))

    test_etr1 = np.load(str(config.SAVE_PATH / "test_preds_etrv3.npy"))
    test_lgb1 = np.load(str(config.SAVE_PATH / "test_preds_lgbv4.npy"))
    test_etr2 = np.load(str(config.SAVE_PATH / "test_preds_etrv4.npy"))
    test_nn1 = np.load(str(config.SAVE_PATH / "test_preds_nnv3.npy"))

    test_preds = 0.5 * test_lgb1 + 0.1 * test_etr1 + 0.1 * test_etr2 + 0.30 * test_nn1.flatten()

    print("Writing submission file")
    test = pd.read_csv(str(config.SAVE_PATH / "testf2.csv"))
    sub = test[["id"]]
    sub["num_orders"] = np.expm1(test_preds)
    sub.to_csv(str(config.SAVE_PATH / "sub_en3v6.csv"), index=False)
    print("All Done")


if __name__ == "__main__":
    main()