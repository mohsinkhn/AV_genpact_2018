import pandas as pd
import numpy as np
import config


def get_grouped_movingmean(df, group_cols, window=60, target_col="num_orders", log=True, method="mean"):
    df = df.copy()
    if log:
        df["target"] = np.log1p(df[target_col])
    else:
        df["target"] = df[target_col]
    cols = "_".join(group_cols)
    name = f"{cols}_{method}{window}"
    if method == "std":
        tmp = df[group_cols + ["target", "week"]].groupby(group_cols)[["target", "week"]].rolling(window=window,
                                                       min_periods=1).std().shift()["target"]
    elif method == "median":
        tmp = df[group_cols + ["target", "week"]].groupby(group_cols)[["target", "week"]].rolling(window=window,
                                                       min_periods=1).median().shift()["target"]
    elif method == "min":
        tmp = df[group_cols + ["target", "week"]].groupby(group_cols)[["target", "week"]].rolling(window=window,
                                                       min_periods=1).min().shift()["target"]
    elif method == "max":
        tmp = df[group_cols + ["target", "week"]].groupby(group_cols)[["target", "week"]].rolling(window=window,
                                                       min_periods=1).max().shift()["target"]
    else:
        tmp = df[group_cols + ["target", "week"]].groupby(group_cols)[["target", "week"]].rolling(window=window,
                                                       min_periods=1).mean().shift()["target"]
    tmp.name = name
    tmp = tmp.reset_index(list(range(len(group_cols))), drop=True)
    print(tmp.head())
    return tmp.to_frame()


def get_valgrouped_stats(df, group_cols, window=60, target_col="num_orders", log=True, method="mean"):
    df = df.copy()
    if log:
        df["target"] = np.log1p(df[target_col])
    else:
        df["target"] = df[target_col]
    cols = "_".join(group_cols)
    name = f"{cols}_{method}{window}"
    max_week = df["week"].max()
    df = df.loc[df["week"] > max_week - window] #Should we put equal to??
    if method == "std":
        tmp = df.groupby(group_cols)["target"].std()
    elif method == "median":
        tmp = df.groupby(group_cols)["target"].median()
    elif method == "min":
        tmp = df.groupby(group_cols)["target"].min()
    elif method == "max":
        tmp = df.groupby(group_cols)["target"].max()
    else:
        tmp = df.groupby(group_cols)["target"].mean()
    tmp.name = name
    return tmp.reset_index()


def get_feats1(tr, val, log=True, target_col="checkout_price"):
    grp_cols_all = [["center_id", "meal_id"],
                    ["center_id"],
                    ["center_id", "category"],
                    ["meal_id"],
                    ["city_code",  "meal_id"]
                    ]

    for grp_cols in grp_cols_all:
        for func in ["mean", "median"]:
            for window in [5, 60, 120]:
                cols = "_".join(grp_cols)
                col_name = f"{cols}_{func}{window}"
                print(f"processing columns {col_name}")
                tr_tmp = get_grouped_movingmean(tr, grp_cols, window=window,
                                                method=func, log=log, target_col=target_col)
                tr = pd.merge(tr, tr_tmp, left_index=True, right_index=True)

                val_tmp = get_valgrouped_stats(tr, grp_cols, window=window,
                                               method=func, log=log, target_col=target_col)
                val = pd.merge(val, val_tmp, on=grp_cols, how="left")
    return tr, val


def main():
    train = pd.read_csv(str(config.SAVE_PATH / "train_all.csv"))
    test = pd.read_csv(str(config.SAVE_PATH / "test_all.csv"))

    tr1 = train.loc[train["week"] < 120]
    val1 = train.loc[train["week"] >= 120]

    tr2 = train.loc[train["week"] < 130]
    val2 = train.loc[train["week"] >= 130]

    tr3 = train.loc[train["week"] < 110]
    val3 = train.loc[train["week"] >= 110]

    tr1, val1 = get_feats1(tr1, val1)
    tr1.to_csv(str(config.SAVE_PATH / "tr1f3.csv"), index=False)
    val1.to_csv(str(config.SAVE_PATH / "val1f3.csv"), index=False)

    tr2, val2 = get_feats1(tr2, val2)
    tr2.to_csv(str(config.SAVE_PATH / "tr2f3.csv"), index=False)
    val2.to_csv(str(config.SAVE_PATH / "val2f3.csv"), index=False)

    tr3, val3 = get_feats1(tr3, val3)
    tr3.to_csv(str(config.SAVE_PATH / "tr3f3.csv"), index=False)
    val3.to_csv(str(config.SAVE_PATH / "val3f3.csv"), index=False)

    train, test = get_feats1(train, test)
    train.to_csv(str(config.SAVE_PATH / "trainf3.csv"), index=False)
    test.to_csv(str(config.SAVE_PATH / "testf3.csv"), index=False)
    print("Saving all files")

if __name__ == "__main__":
    main()