import numpy as np
import pandas as pd
import os
import json
import argparse
from scipy.stats import zscore
import candle
import sys
#sys.path.append('../benchmark-dataset-generator')
import improve_utils


def main():
    split = os.environ['SPLIT'].rstrip('/')
    train_source = os.environ['TRAIN_DATA_SOURCE'].rstrip('/')
    test_source = os.environ['TEST_DATA_SOURCE'].rstrip('/')

    # y_col_name = "auc1"
    y_col_name = "auc"

    process_split(train_source, split, y_col_name)

    if train_source!=test_source:
        process_split(test_source, 'all', y_col_name)


def process_split(data_source, split, y_col_name):
    data_dir = os.environ['CANDLE_DATA_DIR'].rstrip('/')

    try:
        split = int(split)

    except:
        pass

    if isinstance(split, int):
        rs_tr = improve_utils.load_single_drug_response_data_v2(
        source=data_source,
        split_file_name=f"{data_source}_split_{split}_train.txt",
        y_col_name=y_col_name)

        rs_vl = improve_utils.load_single_drug_response_data_v2(
        source=data_source,
        split_file_name=f"{data_source}_split_{split}_val.txt",
        y_col_name=y_col_name)

        rs_te = improve_utils.load_single_drug_response_data_v2(
        source=data_source,
        split_file_name=f"{data_source}_split_{split}_test.txt",
        y_col_name=y_col_name)

        rs_tr.to_csv(data_dir + '/rsp_' + data_source + '_split' + str(split) + '_train.csv')
        rs_vl.to_csv(data_dir + '/rsp_' + data_source + '_split' + str(split) + '_val.csv')
        rs_te.to_csv(data_dir + '/rsp_' + data_source + '_split' + str(split) + '_test.csv')

    else:
        rs_te = improve_utils.load_single_drug_response_data_v2(
        source=data_source,
        split_file_name=f"{data_source}_all.txt",
        y_col_name=y_col_name)

        rs_te.to_csv(data_dir + '/rsp_' + data_source + '_all.csv')


    # Loading cell line expression data
    expression_df = improve_utils.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
    expression_df = expression_df.transpose()
    expression_df.index = [x.strip('ge_') for x in expression_df.index]
    expression_df.index = expression_df.index.rename('Gene_Symbol')

    # Transform expression values into z-score
    expression_df = expression_df.apply(zscore)

    # Loading Gene Sets
    GeneSetFile = data_dir + '/csa_data/raw_data/geneset.gmt'
    GeneSet = []
    GeneSet_Dic = {}

    with open(GeneSetFile) as f:
        for line in f:
            line = line.rstrip().split('\t')
            pathway = line[0]

            # Use only genes present in expression data
            example_cell_line = expression_df.columns[5]
            genes = [x for x in line[2:] if x in expression_df[example_cell_line]]
            GeneSet.extend(genes)
            GeneSet_Dic[pathway] = genes

    GeneSet = set(GeneSet)
    print(str(len(GeneSet)) + ' KEGG genes found in expression file.')

    # Remove genes not present in pathways from expression data
    drop_rows = [x for x in expression_df.index if x not in GeneSet]
    expression_df = expression_df.drop(drop_rows)

    # Save trimmed and normalized expression file and pathway/gene dictionary
    expression_df.to_csv(data_dir + '/ge_' + data_source + '.csv')
    json.dump(GeneSet_Dic, open(data_dir + '/geneset.json', 'w'))

    # Load drug fingerprints file
    drug_fng = improve_utils.load_morgan_fingerprint_data()
    drug_fng.to_csv(data_dir + '/ecfp2_' + data_source + '.csv')


if __name__=="__main__":
    main()

