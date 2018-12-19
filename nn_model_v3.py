import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
import random
import pandas as pd
import numpy as np
import config
import keras
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Flatten, PReLU, Dot
from keras.layers import Input, concatenate, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
tf.set_random_seed(12345786)
random.seed(12345786)
np.random.seed(12345786)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_true, y_preds))


def split_features(X, cont_feats, bin_feats):
    X_list = []
    n = len(X)
    cols = ["week_year", "center_id", "meal_id", "category", "cuisine",
            "center_type", "city_code", "region_code", cont_feats, bin_feats]

    for col in cols:
        X_list.append(X[col].values.reshape((n, -1)))

    return X_list


def nnmodel1():
    # Define our model
    input_weekyear = Input(shape=(1,), name='weekyear')
    em_weekyear = Flatten()(
        Embedding(52, 8, embeddings_initializer=keras.initializers.glorot_uniform(seed=12345786))(input_weekyear))

    input_center_id = Input(shape=(1,), name='center_id')
    em_center_id = Flatten()(
        Embedding(77, 8, embeddings_initializer=keras.initializers.glorot_uniform(seed=12345786))(input_center_id))

    input_meal_id = Input(shape=(1,), name='meal_id')
    em_meal_id = Flatten()(
        Embedding(51, 8, embeddings_initializer=keras.initializers.glorot_uniform(seed=12345786))(input_meal_id))

    input_category = Input(shape=(1,), name='category')
    em_category = Flatten()(
        Embedding(14, 4, embeddings_initializer=keras.initializers.glorot_uniform(seed=12345786))(input_category))

    input_cuisine = Input(shape=(1,), name='cuisine')
    em_cuisine = Flatten()(
        Embedding(4, 3, embeddings_initializer=keras.initializers.glorot_uniform(seed=12345786))(input_cuisine))

    input_ctype = Input(shape=(1,), name='ctype')
    em_ctype = Flatten()(
        Embedding(3, 3, embeddings_initializer=keras.initializers.glorot_uniform(seed=12345786))(input_ctype))

    input_ccode = Input(shape=(1,), name='ccode')
    em_ccode = Flatten()(
        Embedding(51, 8, embeddings_initializer=keras.initializers.glorot_uniform(seed=12345786))(input_ccode))

    input_rcode = Input(shape=(1,), name='rcode')
    em_rcode = Flatten()(
        Embedding(8, 4, embeddings_initializer=keras.initializers.glorot_uniform(seed=12345786))(input_rcode))

    input_cont = Input(shape=(11,), name='cont')
    em_cont = Dense(32, activation='relu',
                    kernel_initializer=keras.initializers.glorot_normal(seed=12345786))(input_cont)
    em_cont = BatchNormalization()(em_cont)

    input_bin = Input(shape=(2,), name='bin')
    em_bin = Dense(2, activation='relu')(input_bin)

    x = concatenate([em_weekyear, em_center_id, em_meal_id, em_category,
                     em_cuisine, em_ctype, em_ccode, em_rcode, em_cont, em_bin
                     ], 1)

    x = Dense(512, activation='relu', kernel_initializer=keras.initializers.glorot_normal(seed=786))(x)

    # x2 = Dense(256, activation='tanh', kernel_initializer=keras.initializers.he_normal(seed=12345786))(x)

    # x3 = Dot(1)([x1, x2])
    # x = concatenate([x1, x2])
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)

    # x = Dense(256,  kernel_initializer=keras.initializers.he_normal(seed=12345786))(x)
    # x = PReLU()(x)
    # x = BatchNormalization()(x)
    # x = Dropout(rate=0.2)(x)

    x = Dense(128, activation='relu', kernel_initializer=keras.initializers.he_normal(seed=786))(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.1)(x)

    main_output = Dense(1, kernel_regularizer=keras.regularizers.l2(1e-5))(x)

    model = Model(inputs=[input_weekyear, input_center_id, input_meal_id, input_category, input_cuisine,
                          input_ctype, input_ccode, input_rcode, input_cont, input_bin],
                  outputs=main_output)

    adam = keras.optimizers.Adam(lr=0.001, decay=1e-7, clipvalue=5)
    model.compile(optimizer=adam, loss=root_mean_squared_error)

    return model


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
            'city_code_meal_id_median120']

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

    tr2["week_year"] = tr2["week"] % 52
    val2["week_year"] = val2["week"] % 52

    train["week_year"] = train["week"] % 52
    test["week_year"] = test["week"] % 52

    tr2["discount"] = (tr2['checkout_price'] - tr2["base_price"]) / tr2["base_price"]
    val2["discount"] = (val2['checkout_price'] - val2["base_price"]) / val2["base_price"]

    train["discount"] = (train['checkout_price'] - train["base_price"]) / train["base_price"]
    test["discount"] = (test['checkout_price'] - test["base_price"]) / test["base_price"]

    cont_feats = ['discount', 'checkout_price', 'base_price',
                  'center_id_meal_id_mean5', 'center_id_meal_id_mean60',
                  'center_id_meal_id_mean120',
                  'center_id_category_mean120',
                  'center_id_meal_id_mean5_price',
                  'center_id_meal_id_mean120_price',
                  'city_code_meal_id_mean5_price',
                  'city_code_meal_id_mean120_price',
                  # 'center_id_meal_id_median5',
                  # 'center_id_meal_id_median60', 'center_id_meal_id_median120',
                  # 'center_id_mean5', 'center_id_mean60',
                  # 'center_id_mean120',
                  # 'center_id_median5', 'center_id_median60', 'center_id_median120',
                  # 'center_id_category_mean5', 'center_id_category_mean60',
                  # 'center_id_category_mean120', #'center_id_category_median5',
                  # 'center_id_category_median60', 'center_id_category_median120',
                  ]
    scaler = StandardScaler()
    tr2[cont_feats] = scaler.fit_transform(tr2[cont_feats])
    val2[cont_feats] = scaler.transform(val2[cont_feats])

    train[cont_feats] = scaler.fit_transform(train[cont_feats])
    test[cont_feats] = scaler.transform(test[cont_feats])

    bin_feats = ['emailer_for_promotion',
                 'homepage_featured']
    scaler = MinMaxScaler(feature_range=(0, 1))
    tr2[bin_feats] = scaler.fit_transform(tr2[bin_feats])
    val2[bin_feats] = scaler.transform(val2[bin_feats])

    train[bin_feats] = scaler.fit_transform(train[bin_feats])
    test[bin_feats] = scaler.transform(test[bin_feats])

    X_tr = split_features(tr2.loc[tr2.week > 1], cont_feats, bin_feats)
    X_val = split_features(val2, cont_feats, bin_feats)

    y_tr = np.log1p(tr2.loc[tr2.week > 1, "num_orders"])
    y_val = np.log1p(val2["num_orders"])

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    y_preds_all = []
    for i in range(5):
        model = nnmodel1()
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=20, batch_size=2048)
        y_preds = model.predict(X_val, batch_size=5000)
        y_preds_all.append(y_preds)

    y_preds_all = np.mean(y_preds_all, axis=0)
    print("Validation RMSE for NN model ", rmse(y_val, y_preds_all))

    X_train = split_features(train.loc[train.week > 1])
    X_test = split_features(test)

    y_train = np.log1p(train.loc[train.week > 1, "num_orders"])

    y_test_all = []
    for i in range(5):
        model = nnmodel1()
        model.fit(X_train, y_train, epochs=20, batch_size=2048)
        y_preds = model.predict(X_test, batch_size=5000)
        y_test_all.append(y_preds)

    y_test_all = np.mean(y_test_all, axis=0)

    np.save(str(config.SAVE_PATH / "val_preds_nnv3.npy"), y_preds_all)
    np.save(str(config.SAVE_PATH / "test_preds_nnv3.npy"), y_test_all)


if __name__ == "__main__":
    main()
