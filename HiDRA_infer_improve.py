# Import basic packages
import numpy as np
import pandas as pd
import os
import sys
import argparse
import json
from improve import framework as frm
from improve.metrics import compute_metrics

# Import keras modules
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate, multiply, dot, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold, train_test_split
from HiDRA_preprocess_improve import preprocess_params
from HiDRA_train_improve import metrics_list, train_params
from pathlib import Path

filepath = Path(__file__).resolve().parent

app_infer_params = []
model_infer_params = []
infer_params = app_infer_params + model_infer_params


def parse_data(ic50, expr, GeneSet_Dic, drugs):
    # Divide expression data based on pathway
    X = []

    for pathway in GeneSet_Dic.keys():
        df = expr[GeneSet_Dic[pathway]]
        X.append(df.loc[ic50['improve_sample_id']])

    X.append(drugs.loc[ic50.index])

    return X


def run(params):
    frm.create_outdir(outdir=params["infer_outdir"])
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    expr = pd.read_csv(params['train_ml_data_dir'] + '/cancer_ge_kegg.csv', index_col=0)
    GeneSet_Dic = json.load(open(params['train_ml_data_dir'] + '/geneset.json', 'r'))
    drugs = pd.read_csv(params['train_ml_data_dir'] + '/drug_ecfp4_nbits512.csv', index_col=0)

    auc_test = pd.read_csv(params['test_ml_data_dir'] + '/test_y_data.csv', index_col=0)
    test_label = auc_test[params['y_col_name']]
    test_input = parse_data(auc_test, expr, GeneSet_Dic, drugs)

    print(params['test_ml_data_dir'] + '/test_y_data.csv')
    print(params["model_outdir"] + '/model.hdf5')
    model = load_model(params["model_outdir"] + '/model.hdf5')

    test_label = test_label.to_numpy().flatten()
    test_pred = model.predict(test_input).flatten()

    frm.store_predictions_df(
        params, y_true=test_label, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"]
    )

    test_scores = frm.compute_performace_scores(
        params, y_true=test_label, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores


def main(args):
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="improve_hidra_default_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )

    test_scores = run(params)
    print("\nFinished inference with HiDRA.")


if __name__ == "__main__":
    main(sys.argv[1:])
