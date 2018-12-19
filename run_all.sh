#!/usr/bin/env bash
python prepare_train_test_files.py >> prepare_data.log
python generate_features2.py >> gen_features2.log
python generate_features3.py >> gen_features3.log
python model_8.py >> etr_v3.log
python model_10.py >> lgb_etr_v4.log
python nn_model_v3.py >> nn_model.log
python stacker_v4.py >> stacker.log