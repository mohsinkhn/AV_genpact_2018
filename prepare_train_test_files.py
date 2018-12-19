import pandas as pd

import config
"""
Join all data sources
Label encode categoricals
"""


def main():
    # Read train and test files
    train = pd.read_csv(str(config.TRAIN_FILE))
    test = pd.read_csv(str(config.TEST_FILE))

    # merge data
    all_data = pd.concat([train, test], sort=False).reset_index(drop=True)

    # join meal and center info
    meal_info = pd.read_csv(str(config.MEAL_INFO_FILE))
    center_info = pd.read_csv(str(config.CENTER_INFO_FILE))
    all_data = pd.merge(all_data, meal_info, on=["meal_id"], how="left")
    all_data = pd.merge(all_data, center_info, on=["center_id"], how="left")
    all_data["category"] = all_data["category"].astype("category").cat.codes
    all_data["cuisine"] = all_data["cuisine"].astype("category").cat.codes
    all_data["city_code"] = all_data["city_code"].astype("category").cat.codes
    all_data["region_code"] = all_data["region_code"].astype("category").cat.codes
    all_data["center_type"] = all_data["center_type"].astype("category").cat.codes
    all_data = all_data.sort_values(by=["center_id", "meal_id", "week"])

    # split and save
    train = all_data.loc[all_data["week"] < 146]
    test = all_data.loc[all_data["week"] >= 146]

    print("final train an test shapes are ", train.shape, test.shape)
    train.to_csv(str(config.SAVE_PATH / "train_all.csv"), index=False)
    test.to_csv(str(config.SAVE_PATH / "test_all.csv"), index=False)


if __name__ == "__main__":
    main()
