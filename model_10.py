import pandas as pd
import numpy as np
import lightgbm as lgb
import config
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
np.random.seed(786)


def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_true, y_preds))


def get_val(tr, val, feats, lgb_params):
    X_tr = tr.loc[tr.week > 1, feats]
    X_val = val[feats]
    y_tr = np.log1p(tr.loc[tr.week > 1, "num_orders"])
    y_val = np.log1p(val["num_orders"])

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr,
              eval_set=[
                  (X_tr, y_tr),
                  (X_val, y_val)],
              eval_metric='rmse',
              verbose=200,  # early_stopping_rounds=600,
              )
    val_preds = model.predict(X_val)
    return val_preds


def get_test(tr, test, feats, lgb_params):
    X_tr = tr.loc[tr.week > 1, feats]
    X_test = test[feats]
    y_tr = np.log1p(tr.loc[tr.week > 1, "num_orders"])

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr,
              eval_set=[
                  (X_tr, y_tr),
              ],
              eval_metric='rmse',
              verbose=200,  # early_stopping_rounds=600,
              )
    test_preds = model.predict(X_test)
    return test_preds


def main():
    cols = ['id', 'center_id_meal_id_mean5', 'center_id_meal_id_mean60',
            'center_id_meal_id_mean120', 'center_id_meal_id_median5',
            'center_id_meal_id_median60', 'center_id_meal_id_median120',
            'center_id_mean5', 'center_id_mean60', 'center_id_mean120',
            'center_id_median5', 'center_id_median60', 'center_id_median120',
            'center_id_category_mean5', 'center_id_category_mean60',
            'center_id_category_mean120', 'center_id_category_median5',
            'center_id_category_median60', 'center_id_category_median120',
            'meal_id_mean5', 'meal_id_mean60', 'meal_id_mean120', 'meal_id_median5',
            'meal_id_median60', 'meal_id_median120', 'city_code_meal_id_mean5',
            'city_code_meal_id_mean60', 'city_code_meal_id_mean120',
            'city_code_meal_id_median5', 'city_code_meal_id_median60',
            'city_code_meal_id_median120'
            ]

    tr2 = pd.read_csv(str(config.SAVE_PATH / "tr2f2.csv"))
    val2 = pd.read_csv(str(config.SAVE_PATH / "val2f2.csv"))

    tr22 = pd.read_csv(str(config.SAVE_PATH / "tr2f3.csv"), usecols=cols)
    val22 = pd.read_csv(str(config.SAVE_PATH / "val2f3.csv"), usecols=cols)

    train = pd.read_csv(str(config.SAVE_PATH / "trainf2.csv"))
    test = pd.read_csv(str(config.SAVE_PATH / "testf2.csv"))

    train2 = pd.read_csv(str(config.SAVE_PATH / "trainf3.csv"), usecols=cols)
    test2 = pd.read_csv(str(config.SAVE_PATH / "testf3.csv"), usecols=cols)

    cols_new = ['id'] + [col + "_price" for col in cols if col != "id"]

    tr22.columns = cols_new
    val22.columns = cols_new
    train2.columns = cols_new
    test2.columns = cols_new

    tr2 = pd.merge(tr2, tr22, on="id")
    val2 = pd.merge(val2, val22, on="id")
    train = pd.merge(train, train2, on="id")
    test = pd.merge(test, test2, on="id")
    print(tr2.shape, val2.shape, train.shape, test.shape)

    tr2["week_year"] = tr2["week"] % 52
    val2["week_year"] = val2["week"] % 52

    train["week_year"] = train["week"] % 52
    test["week_year"] = test["week"] % 52

    tr2["discount"] = (tr2['checkout_price'] - tr2["base_price"]) / tr2["base_price"]
    val2["discount"] = (val2['checkout_price'] - val2["base_price"]) / val2["base_price"]

    train["discount"] = (train['checkout_price'] - train["base_price"]) / train["base_price"]
    test["discount"] = (test['checkout_price'] - test["base_price"]) / test["base_price"]

    tmp = pd.concat([train, test]).groupby(["center_id"])["week"].first()
    tr2["week_start"] = tr2.center_id.map(tmp)
    val2["week_start"] = val2.center_id.map(tmp)
    train["week_start"] = train.center_id.map(tmp)
    test["week_start"] = test.center_id.map(tmp)

    tmp = pd.concat([train, test]).set_index("id").groupby(["center_id"])["week"].diff().reset_index(0, drop=True)
    tr2["week_gap"] = tr2.id.map(tmp)
    val2["week_gap"] = val2.id.map(tmp)
    train["week_gap"] = train.id.map(tmp)
    test["week_gap"] = test.id.map(tmp)

    tmp = tr2.set_index("id").groupby(["center_id", "meal_id"])["num_orders"].shift()
    tr2["last_order"] = tr2.id.map(tmp)
    tmp = tr2.groupby(["center_id", "meal_id"])["num_orders"].last()
    tmp.name = "last_order"
    val2 = val2.join(tmp, on=["center_id", "meal_id"])
    tmp = train.set_index("id").groupby(["center_id", "meal_id"])["num_orders"].shift()
    train["last_order"] = train.id.map(tmp)
    tmp = train.groupby(["center_id", "meal_id"])["num_orders"].last()
    tmp.name = "last_order"
    test = test.join(tmp, on=["center_id", "meal_id"])

    tmp = pd.concat([train, test]).set_index("id").groupby(["center_id", "meal_id"])["base_price"].diff()
    tr2["price_diff"] = tr2.id.map(tmp) / tr2["center_id_meal_id_mean120_price"]
    val2["price_diff"] = val2.id.map(tmp) / val2["center_id_meal_id_mean120_price"]
    train["price_diff"] = train.id.map(tmp) / train["center_id_meal_id_mean120_price"]
    test["price_diff"] = test.id.map(tmp) / test["center_id_meal_id_mean120_price"]

    tr2["disc_rat"] = tr2["last_order"] / tr2["base_price"]
    val2["disc_rat"] = val2["last_order"] / val2["base_price"]
    train["disc_rat"] = train["last_order"] / train["base_price"]
    test["disc_rat"] = test["last_order"] / test["base_price"]

    feats = ["center_id", "meal_id",
             'checkout_price', 'base_price', 'emailer_for_promotion',
             'homepage_featured',
             'category', 'cuisine',
             'region_code', 'center_type',
             'week_year', 'discount', 'op_area',
             'center_id_meal_id_mean120',
             'center_id_meal_id_mean5',
             # 'center_id_meal_id_mean60',
             'center_id_category_mean120',
             'center_id_meal_id_median120',
             'center_id_meal_id_mean5_price',
             'center_id_meal_id_mean120_price',
             'city_code_meal_id_mean5_price',
             'city_code_meal_id_mean120_price',
             'week_start',
             'week_gap',
             'last_order',
             'price_diff',
             'disc_rat',
             ]

    lgb_params = {
        'device': 'cpu',
        'boosting_type': 'gbdt',
        # 'xgboost_mode': True,
        'n_jobs': -1,
        'max_depth': -1,
        'n_estimators': 10000,
        'colsample_bytree': 0.5,
        'learning_rate': 0.05,
        'min_child_samples': 20,
        'min_child_weight': 1,
        'num_leaves': 7,
        'subsample': 0.9,
        'reg_lambda': 0.01,
        'reg_alpha': 0.0,
        'random_state': 12345786,
    }

    val_preds_all = []
    for seed in [12345786, 111, 786]:
        lgb_params["random_state"] = seed
        val_preds = get_val(tr2, val2, feats, lgb_params)
        val_preds_all.append(val_preds)

    val_preds_out = np.mean(val_preds_all, axis=0)
    print("Validation RMSE is", rmse(np.log1p(val2["num_orders"]), val_preds_out))

    test_preds_lgb = []
    for seed in [12345786, 111, 786]:
        lgb_params["random_state"] = seed
        lgb_params["n_estimators"] = 11200
        test_preds = get_test(train, test, feats, lgb_params)
        test_preds_lgb.append(test_preds)
    test_preds_lgb = np.mean(test_preds_lgb, axis=0)

    print("Saving test preds")
    np.save(str(config.SAVE_PATH / "val_preds_lgbv4.npy"), val_preds_out)
    np.save(str(config.SAVE_PATH / "test_preds_lgbv4.npy"), test_preds_lgb)
    print("Lightgbm model v4 done!")

    etr = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, min_samples_split=25, random_state=786)
    etr_preds = etr.fit(tr2[feats].fillna(-99), np.log1p(tr2["num_orders"])).predict(val2[feats].fillna(-99))
    test_preds_etr = etr.fit(train[feats].fillna(-99), np.log1p(train["num_orders"])).predict(test[feats].fillna(-99))
    print("Validation RMSE ousin ETR is", rmse(np.log1p(val2["num_orders"]), etr_preds))
    np.save(str(config.SAVE_PATH / "val_preds_etrv4.npy"), etr_preds)
    np.save(str(config.SAVE_PATH  / "test_preds_etrv4.npy"), test_preds_etr)
    print("ETR model v4 done!")


if __name__ == "__main__":
    main()

