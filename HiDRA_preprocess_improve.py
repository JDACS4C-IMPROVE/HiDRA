import numpy as np
import pandas as pd
import os
import json
import argparse
from scipy.stats import zscore
import sys
from pathlib import Path
from improve import framework as frm
from improve import drug_resp_pred as drp


filepath = Path(__file__).resolve().parent
improve_data_dir = os.environ["IMPROVE_DATA_DIR"].rstrip('/')

# Model-specific params
model_preproc_params = [
    {"name": "kegg_pathway_file",
     "type": str,
     "help": "KEGG file of pathways and genes",
    },
]

# App-specific params (App: drug response prediction)
app_preproc_params = [
    {"name": "x_data_canc_files",
     "type": str,
     "help": "List of feature files.",
    },
    {"name": "x_data_drug_files",
     "type": str,
     "help": "List of feature files.",
    },
    {"name": "y_data_files",
     "type": str,
     "help": "List of output files.",
    },
    {"name": "canc_col_name",
     "default": "improve_sample_id",
     "type": str,
     "help": "Column name that contains the cancer sample ids.",
    },
    {"name": "drug_col_name",
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name that contains the drug ids.",
    },
]

preprocess_params = model_preproc_params + app_preproc_params


def gene_selection(df, genes_fpath, canc_col_name):
    """
    Read in a KEGG pathway list and keep only genes present in a pathway and
    gene expression data
    """
    GeneSet = []
    GeneSet_Dic = {}
    df_genes = set([x for x in df.columns])

    with open(genes_fpath) as f:
        for line in f:
            line = line.rstrip().split('\t')
            pathway = line[0]

            genes = [x for x in line[2:] if x in df_genes]
            GeneSet.extend(genes)
            GeneSet_Dic[pathway] = genes

    GeneSet = set(GeneSet)
    print(str(len(GeneSet)) + ' KEGG genes found in expression file.')
    genes = drp.common_elements(GeneSet, df.columns[1:])
    cols = [canc_col_name] + genes

    return df[cols], GeneSet_Dic


def run(params):
    """ Execute data pre-processing for HiDRA.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.
    """
    params = frm.build_paths(params)
    processed_outdir = frm.create_outdir(params['ml_data_outdir'])
    print("\nLoading omics data...")
    oo = drp.OmicsLoader(params)

    ge = oo.dfs['cancer_gene_expression.tsv']

    genes_fpath = improve_data_dir + '/' + params["kegg_pathway_file"]
    ge, GeneSet_Dic = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])

    json.dump(GeneSet_Dic, open(processed_outdir/'geneset.json', 'w'))

    # Check that z-score is on the correct axis
    numeric_cols = ge.select_dtypes(include=[np.number]).columns
    ge[numeric_cols] = ge[numeric_cols].apply(zscore, axis=1)

    print("\nLoading drugs data...")
    dd = drp.DrugsLoader(params)

    dr = dd.dfs['drug_ecfp4_nbits512.tsv']

    ge.to_csv(processed_outdir/'cancer_ge_kegg.csv', index=False)
    dr.to_csv(processed_outdir/'drug_ecfp4_nbits512.csv')

    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():
        rr = drp.DrugResponseLoader(params, split_file=split_file, verbose=True)
        df_response = rr.dfs[params['y_data_files'][0][0]]

        df_y, df_canc = drp.get_common_samples(df1=df_response, df2=ge,
                                               ref_col=params["canc_col_name"])

        df_y = df_y[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]

        data_fname = frm.build_ml_data_name(params, stage)

        y_data_fname = f"{stage}_{params['y_data_suffix']}.csv"
        df_y.to_csv(processed_outdir / y_data_fname, index=False)

    return processed_outdir


def main():
    additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        default_model="improve_hidra_default_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    processed_outdir = run(params)
    print("\nFinished HiDRA pre-processing.")


if __name__=="__main__":
    main()
