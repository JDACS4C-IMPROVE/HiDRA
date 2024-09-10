"""
Model-specific params (Model: HiDRA)
If no params are required by the model, then it should be an empty list.
"""

from improvelib.utils import str2bool


preprocess_params = [
    {"name": "kegg_pathway_file",
     "type": str,
     "help": "File of KEGG pathways to structure data and model.",
    },
]

train_params = []

infer_params = []
