## Overview
HiDRA (Hierarchical Network for Drug Response Prediction with Attention) is a drug response prediction network [published](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00706) by Iljung Jim and Hojung Nam (2021). Here we have brought it into the IMPROVE framework.

HiDRA consists of four smaller networks: a drug feature encoding network, which takes SMILES strings converted to Morgan fingerprints; a set of gene-level networks which encode expression data for genes in each pathway; a pathway-level network that takes in output of the individual gene-level networks; and a prediction network that uses drug encodings and pathway output to predict ln(IC50). Each sub-network consists of two dense layers and an attention module (tanh + softmax activation).

## Dependencies
- CANDLE-ECP (develop branch)
- tensorflow-gpu (2.4.2)
- scikit-learn (0.24.2)
- pandas (1.1.5)
- openpyxl (3.0.9)

## Setup

After cloning this repository, setup the environment with
```
conda env create -f environment.yml
conda activate hidra
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```

## Data

All data must be placed within a directory `csa_data`.

Four files are required to run HiDRA:
    - A KEGG pathway file, where each line is a pathway name followed by a list of gene symbols
    - A gene expression file with gene symbols as rows and cell lines as columns. Ideally this file will contain all or most genes present in the KEGG pathway file
    - A 512-bit Morgan fingerprint drug file
    - A response table of cancer/drug pairs

To use the IMPROVE benchmark dataset, the user must set up the directory structure
```
mkdir csa_data
mkdir csa_data/raw_data
mkdir csa_data/raw_data/y_data
mkdir csa_data/raw_data/x_data
mkdir csa_data/raw_data/splits
```

Data must then be downloaded from the IMPROVE FTP with `wget`.

```
wget -P csa_data/raw_data/y_data https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/y_data/response.tsv

wget -P csa_data/raw_data/x_data https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/x_data/cancer_gene_expression.tsv

wget -P csa_data/raw_data/x_data https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/x_data/drug_ecfp4_nbits512.tsv

wget -P csa_data/raw_data https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/hidra/raw_data/geneset.gmt
```

If we want to train on CCLE training data and test on CCLE testing data we also download the split files:
```
wget -P csa_data/raw_data/splits https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/splits/CCLE_split_0_train.txt

wget -P csa_data/raw_data/splits https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/splits/CCLE_split_0_val.txt

wget -P csa_data/raw_data/splits https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/splits/CCLE_split_0_test.txt
```

## Creating Models

To run training and testing, the user must specify a device (GPU) to use and a directory to store processed data and output. If we want to train on GPU 3 we run:
```
CUDA_VISIBLE_DEVICES=3 CANDLE_DATA_DIR=my_data_dir SPLIT=0 TRAIN_DATA_SOURCE=CCLE TEST_DATA_SOURCE=CCLE python csa_feature_gen.py

CUDA_VISIBLE_DEVICES=3 CANDLE_DATA_DIR=my_data_dir SPLIT=0 TRAIN_DATA_SOURCE=CCLE TEST_DATA_SOURCE=CCLE python csa_training.py

CUDA_VISIBLE_DEVICES=3 CANDLE_DATA_DIR=my_data_dir SPLIT=0 TRAIN_DATA_SOURCE=CCLE TEST_DATA_SOURCE=CCLE python csa_predict.py
```

Training hyperparameters are set in `hidra_default_model.txt`.

## Data Preprocessing Steps

Data preprocessing includes the following steps:
- Download gene expression, drug fingerprint, and response pair data from FTP site
- Transform gene expressions to z-scores
- Load KEGG pathway data and remove all genes from the expression file not present in a pathway
- Save KEGG pathways as a JSON file
- Load response data and remove unnecessary columns
	
If using the original GDSC1000 data, a few extra reformatting steps are needed:
- Load expression dataset, remove genes without valid symbols, and change COSMIC identifiers to sample names to be consistent with response data
- Load response data, reformat pivot table into a list of pairs, and remove any entries with no fingerprint or expression data

preprocess.sh, train.sh, and infer.sh set which GPU to use, set the data directory, and run preprocessing, training, and testing scripts, respectively. 
