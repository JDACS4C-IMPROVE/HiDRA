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
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
import improvelib.applications.drug_response_prediction.drp_utils as drp

# Model-specifc imports
from model_params_def import preprocess_params
from hidra_utils import *


filepath = Path(__file__).resolve().parent


def run(params: Dict):
    """ Run data preprocessing.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the preprocessed (generated)
            ML data files.
    """

    print("\nLoading omics data...")
    omics_obj = omics_utils.OmicsLoader(params)
    ge = omics_obj.dfs['cancer_gene_expression.tsv']

    genes_fpath = params["input_supp_data_dir"] + '/geneset.gmt'
    ge, GeneSet_Dic = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])

    json.dump(GeneSet_Dic, open(params['output_dir'] + '/geneset.json', 'w'))

    # Check that z-score is on the correct axis
    numeric_cols = ge.select_dtypes(include=[np.number]).columns
    ge[numeric_cols] = ge[numeric_cols].apply(zscore, axis=1)

    print("\nLoading drugs data...")
    drugs_obj = drugs_utils.DrugsLoader(params)
    mf = drugs_obj.dfs['drug_ecfp4_nbits512.tsv']
    mf = mf.reset_index()

    ge.to_csv(params["output_dir"] + '/cancer_ge_kegg.csv', index=False)
    mf.to_csv(params["output_dir"] + '/drug_ecfp4_nbits512.csv', index=False)

    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():
        rsp = drp.DrugResponseLoader(params,
                                     split_file=split_file,
                                     verbose=False).dfs["response.tsv"]

#        rsp = rsp.merge(ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
#        rsp = rsp.merge(mf[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
#        ge_sub = ge[ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])].reset_index(drop=True)
#        mf_sub = mf[mf[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])].reset_index(drop=True)

        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)

#        print("Merge data")
#        data = rsp.merge(ge_sc, on=params["canc_col_name"], how="inner")
#        data = data.merge(md_sc, on=params["drug_col_name"], how="inner")
#        data = data.sample(frac=1.0).reset_index(drop=True) # shuffle

        print("Save data")
#        data = data.drop(columns=["study"]) # to_parquet() throws error since "study" contain mixed values
#        data.to_parquet(Path(params["output_dir"]) / data_fname) # saves ML data file to parquet

        # Prepare the y dataframe for the current stage
#        fea_list = ["ge", "mordred"]
#        fea_cols = [c for c in data.columns if (c.split(fea_sep)[0]) in fea_list]
#        meta_cols = [c for c in data.columns if (c.split(fea_sep)[0]) not in fea_list]
#        ydf = data[meta_cols]
        ydf = rsp

        # [Req] Save y dataframe for the current stage
        frm.save_stage_ydf(ydf, stage, params["output_dir"])

    return params["output_dir"]


def main(args):
    additional_definitions = preprocess_params
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="hidra_params.txt",
        additional_definitions=additional_definitions,
    )
    ml_data_outdir = run(params)
    print("\nFinished HiDRA pre-processing.")


if __name__=="__main__":
    main(sys.argv[1:])
