import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import json
from hidra_utils import *

# Import improvelib
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from model_params_def import infer_params

filepath = Path(__file__).resolve().parent


def run(params):
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")

    dir = Path(params["input_data_dir"])
    expr = pd.read_csv(dir / 'cancer_ge_kegg.csv', index_col=0)
    GeneSet_Dic = json.load(open(dir / 'geneset.json', 'r'))
    drugs = pd.read_csv(dir / 'drug_ecfp4_nbits512.csv', index_col=0)
    auc_test = pd.read_csv(dir / 'test_y_data.csv', index_col=0)

    test_label = auc_test[params['y_col_name']]
    test_input = parse_data(auc_test, expr, GeneSet_Dic, drugs)

    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"])

    model = load_model(str(modelpath))

    test_label = test_label.to_numpy().flatten()
    test_pred = model.predict(test_input).flatten()

    frm.store_predictions_df(
        y_true=test_label, y_pred=test_pred, stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"]
    )

    print(params["calc_infer_scores"])

    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=test_label,
            y_pred=test_pred,
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


def main(args):
    additional_definitions = infer_params
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="hidra_params.txt",
        additional_definitions=additional_definitions
    )
    status = run(params)
    print("\nFinished inference with HiDRA.")


if __name__ == "__main__":
    main(sys.argv[1:])
