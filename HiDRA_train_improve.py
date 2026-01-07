import sys
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import json
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics
from model_params_def import train_params
from hidra_utils import *

filepath = Path(__file__).resolve().parent

metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]


def run(params: Dict):
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"],
                                     model_file_format=params["model_file_format"],
                                     model_dir=params["output_dir"])
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")

    # Load data
    dir = Path(params["input_dir"])
    expr = pd.read_csv(dir / 'cancer_ge_kegg.csv', index_col=0)
    GeneSet_Dic = json.load(open(dir / 'geneset.json', 'r'))
    drugs = pd.read_csv(dir / 'drug_ecfp4_nbits512.csv', index_col=0)
    auc_tr = pd.read_csv(dir / 'train_y_data.csv')
    auc_val = pd.read_csv(dir / 'val_y_data.csv')

    # Train
    batch_size = params['batch_size']
    epochs = params['epochs']
    loss = params['loss']
    lr = params['learning_rate']
    y_name = params['y_col_name']

    train_label = auc_tr[y_name]
    val_label = auc_val[y_name]

#    train_input = parse_data(auc_tr, expr, GeneSet_Dic, drugs)
    val_input = parse_data(auc_val, expr, GeneSet_Dic, drugs)
    shuffle = True
    train_gen = MultiGenerator(expr, drugs, GeneSet_Dic, auc_tr, batch_size, y_name, shuffle)
    val_gen = MultiGenerator(expr, drugs, GeneSet_Dic, auc_val, batch_size, y_name, shuffle)

    model_saver = ModelCheckpoint(str(modelpath), monitor='val_loss',
                                  save_best_only=True, save_weights_only=False)

    model_stopper = EarlyStopping(monitor='val_loss', restore_best_weights=True,
                                  patience=params['patience'])

    callbacks = [model_saver, model_stopper]
    optimizer = Adam(learning_rate=lr)

    model = Making_Model(GeneSet_Dic)
    model.compile(loss=loss, optimizer=optimizer)
#    history = model.fit(train_input, train_label, shuffle=True,
#                     epochs=epochs, batch_size=batch_size, verbose=2,
#                     validation_data=(val_input,val_label),
#                     callbacks=callbacks)

    history = model.fit(train_gen, validation_data=val_gen,
                        epochs=epochs, verbose=2, callbacks=callbacks)

    model.save(str(modelpath))

    # Make predictions
    val_label = val_label.to_numpy().flatten()
    val_pred = model.predict(val_input).flatten()

    frm.store_predictions_df(
        y_true=val_label, y_pred=val_pred, stage="val",
        y_col_name=params["y_col_name"], output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    val_scores = frm.compute_performance_scores(
        y_true=val_label, y_pred=val_pred, stage="val",
        metric_type=params["metric_type"], output_dir=params["output_dir"]
    )

    val_loss = np.min(history.history['val_loss'])

    print('IMPROVE_RESULT val_loss:\t' + str(val_loss))

    return val_scores


def main(args):
    additional_definitions = train_params
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="hidra_params.txt",
        additional_definitions=additional_definitions)
    val_scores = run(params)
    print("\nFinished training HiDRA model.")


if __name__ == "__main__":
    main(sys.argv[1:])

    try:
        K.clear_session()

    except AttributeError:
        pass
