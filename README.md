# HiDRA

This repository demonstrates how to use the [IMPROVE library v0.1.0-2024-09-27](https://jdacs4c-improve.github.io/docs/) for building a drug response prediction (DRP) model using HiDRA (Hierarchical Network for Drug Response Prediction with Attention), and provides examples with the benchmark [cross-study analysis (CSA) dataset](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

This version, tagged as `v0.1.0-2024-09-27`, , introduces a new API which is designed to encourage broader adoption of IMPROVE and its curated models by the research community.


## Dependencies
Installation instuctions are detailed below in [Step-by-step instructions](#step-by-step-instructions).

Conda `environment.yml` file

ML framework:
+ [Tensorflow](https://github.com/tensorflow/docs) - machine learning framework for building the model
+ [scikit-learn](https://github.com/scikit-learn/scikit-learn) - train/val/test split

IMPROVE dependencies:
+ [IMPROVE v0.1.0-2024-09-27](https://jdacs4c-improve.github.io/docs/)


## Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

The data tree is shown below:
```
csa_data/raw_data/
├── splits
│   ├── CCLE_all.txt
│   ├── CCLE_split_0_test.txt
│   ├── CCLE_split_0_train.txt
│   ├── CCLE_split_0_val.txt
│   ├── CCLE_split_1_test.txt
│   ├── CCLE_split_1_train.txt
│   ├── CCLE_split_1_val.txt
│   ├── ...
│   ├── GDSCv2_split_9_test.txt
│   ├── GDSCv2_split_9_train.txt
│   └── GDSCv2_split_9_val.txt
├── x_data
│   ├── cancer_copy_number.tsv
│   ├── cancer_discretized_copy_number.tsv
│   ├── cancer_DNA_methylation.tsv
│   ├── cancer_gene_expression.tsv
│   ├── cancer_miRNA_expression.tsv
│   ├── cancer_mutation_count.tsv
│   ├── cancer_mutation_long_format.tsv
│   ├── cancer_mutation.parquet
│   ├── cancer_RPPA.tsv
│   ├── drug_ecfp4_nbits512.tsv
│   ├── drug_info.tsv
│   ├── drug_mordred_descriptor.tsv
│   └── drug_SMILES.tsv
└── y_data
    └── response.tsv
```


## Model scripts and parameter file
+ `HiDRA_preprocess_improve.py` - takes benchmark data files and transforms into files for training and inference
+ `HiDRA_train_improve.py` - trains the HiDRA DRP model
+ `HiDRA_infer_improve.py` - runs inference with the trained HiDRA model
+ `hidra_utils.py` - defines the structure of the model and creates data generators
+ `hidra_params.txt` - default parameter file



# Step-by-step instructions

### 1. Clone the model repository
```
git clone git@github.com:JDACS4C-IMPROVE/HiDRA.git
cd HiDRA
```

### 2. Set computational environment
Create conda env using yml file
```
conda env create -f environment.yml 
```


### 3. Run `setup_improve.sh`.
```
source setup_improve.sh
```

This will:
1. Download cross-study analysis (CSA) benchmark data into `./csa_data/`.
2. Clone IMPROVE repo outside the HiDRA model repo
3. Set up `PYTHONPATH` (adds IMPROVE repo).


### 4. Preprocess CSA benchmark data (_raw data_) to construct model input data (_ML data_)
```
python HiDRA_preprocess_improve.py
```

Preprocesses the CSA data and creates train, validation (val), and test datasets.

Generates:
* three model input data files: `cancer_ge_kegg.csv`, `drug_ecfp4_nbits512.csv`, `geneset.json`
* three tabular data files, each containing the drug response values (i.e. AUC) and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

```
exp_result
├── cancer_ge_kegg.csv
├── drug_ecfp4_nbits512.csv
├── geneset.json
├── test_y_data.csv
├── train_y_data.csv
└── val_y_data.csv
```


### 5. Train HiDRA model
```
python HiDRA_train_improve.py
```

Trains the HiDRA model using the model input data.

Generates:
* trained model: `model.h5`
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`
```
exp_result
├── cancer_ge_kegg.csv
├── drug_ecfp4_nbits512.csv
├── geneset.json
├── model.h5
├── param_log_file.txt
├── test_y_data.csv
├── train_y_data.csv
├── val_scores.json
├── val_y_data.csv
└── val_y_data_predicted.csv
```


### 6. Run inference on test data with the trained HiDRA model
```
python HiDRA_infer_improve.py
```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
```
exp_result
├── cancer_ge_kegg.csv
├── drug_ecfp4_nbits512.csv
├── geneset.json
├── model.h5    
├── param_log_file.txt
├── test_y_data.csv
├── test_scores.json
├── test_y_data_predicted.csv
├── train_y_data.csv
├── val_scores.json 
├── val_y_data.csv
└── val_y_data_predicted.csv
```
