import numpy as np
import pandas as pd
import sys
import json
from scipy.stats import zscore
from pathlib import Path
from typing import Dict

# Core improvelib imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# Application-specific (DRP) imports
import improvelib.applications.drug_response_prediction.drp_utils as drp

# Model-specifc imports
from model_params_def import preprocess_params
from hidra_utils import *


filepath = Path(__file__).resolve().parent


def run(params: Dict):
    print("\nLoading omics data...")
    ge = frm.get_x_data(file = params['cell_transcriptomic_file'], 
                        benchmark_dir = params['input_dir'], 
                        column_name = params['canc_col_name'])
    ge = ge.reset_index()
    genes_fpath = str(filepath) + '/raw_data/geneset.gmt'
    ge, GeneSet_Dic = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])

    json.dump(GeneSet_Dic, open(params['output_dir'] + '/geneset.json', 'w'))

    # Check that z-score is on the correct axis
    numeric_cols = ge.select_dtypes(include=[np.number]).columns
    ge[numeric_cols] = ge[numeric_cols].apply(zscore, axis=1)

    print("\nLoading drugs data...")
    mf = frm.get_x_data(file = params['drug_ecfp_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])
    mf = mf.reset_index()

    ge.to_csv(params["output_dir"] + '/cancer_ge_kegg.csv', index=False)
    mf.to_csv(params["output_dir"] + '/drug_ecfp4_nbits512.csv', index=False)

    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():
        rsp = frm.get_y_data(split_file=split_file, 
                             benchmark_dir=params['input_dir'], 
                             y_data_file=params['y_data_file'])
        rsp = rsp.dropna(subset=[params['y_col_name']])

        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)
        print("Save data")
        ydf = rsp

        # [Req] Save y dataframe for the current stage
        frm.save_stage_ydf(ydf, stage, params["output_dir"])

    return params["output_dir"]


def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="HiDRA_params.ini",
                                       additional_definitions=preprocess_params)
    timer_preprocess = frm.Timer()
    ml_data_outdir = run(params)
    timer_preprocess.save_timer(dir_to_save=params["output_dir"], 
                                filename='runtime_preprocess.json', 
                                extra_dict={"stage": "preprocess"})
    print("\nFinished HiDRA pre-processing.")


if __name__=="__main__":
    main(sys.argv[1:])
