#!/usr/bin/env bash
python prepare_train_test_files.py
python generate_features2.py
python generate_features3.py
python model_8.py
python model_10.py
python nn_model_v3.py
python stacker_v4.py
