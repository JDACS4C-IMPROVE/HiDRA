# Import basic packages
import numpy as np
import pandas as pd
import os
import sys
import argparse
import json
from improve import framework as frm
from improve.metrics import compute_metrics
#import candle
from HiDRA_preprocess_improve import preprocess_params

# Import keras modules
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate, multiply, dot, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold, train_test_split

file_path = os.path.dirname(os.path.realpath(__file__))

metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]

app_train_params = []

model_train_params = [
    {"name": "epochs",
     "type": int,
     "default": 20,
     "help": "Number of epochs for training."
    },

    {"name": "learning_rate",
     "type": float,
     "default": 0.001,
     "help": "Learning rate for the optimizer."
    },

    {"name": "batch_size",
     "type": int,
     "default": 1024,
     "help": "Training batch size."
    },

    {"name": "loss",
     "type": str,
     "default": "mse",
     "help": "Learning rate for the optimizer."
    },
]

train_params = app_train_params + model_train_params


def parse_data(ic50, expr, GeneSet_Dic, drugs):
    # Divide expression data based on pathway
    X = []

    for pathway in GeneSet_Dic.keys():
        df = expr[GeneSet_Dic[pathway]]
        X.append(df.loc[ic50['improve_sample_id']])

    X.append(drugs.loc[ic50.index])

    return X


def Making_Model(GeneSet_Dic):
    # HiDRA model with keras
    # Drug-level network
    Drug_feature_length = 512
    Drug_Input = Input((Drug_feature_length,), dtype='float32', name='Drug_Input')

    Drug_Dense1 = Dense(256, name='Drug_Dense_1')(Drug_Input)
    Drug_Dense1 = BatchNormalization(name='Drug_Batch_1')(Drug_Dense1)
    Drug_Dense1 = Activation('relu', name='Drug_RELU_1')(Drug_Dense1)

    Drug_Dense2 = Dense(128, name='Drug_Dense_2')(Drug_Dense1)
    Drug_Dense2 = BatchNormalization(name='Drug_Batch_2')(Drug_Dense2)
    Drug_Dense2 = Activation('relu', name='Drug_RELU_2')(Drug_Dense2)

    # Drug network that will be used to attention network in the Gene-level network and Pathway-level network
    Drug_Dense_New1 = Dense(128, name='Drug_Dense_New1')(Drug_Input)
    Drug_Dense_New1 = BatchNormalization(name='Drug_Batch_New1')(Drug_Dense_New1)
    Drug_Dense_New1 = Activation('relu', name='Drug_RELU_New1')(Drug_Dense_New1)

    Drug_Dense_New2 = Dense(32, name='Drug_Dense_New2')(Drug_Dense_New1)
    Drug_Dense_New2 = BatchNormalization(name='Drug_Batch_New2')(Drug_Dense_New2)
    Drug_Dense_New2 = Activation('relu', name='Drug_RELU_New2')(Drug_Dense_New2)

    #Gene-level network
    GeneSet_Model=[]
    GeneSet_Input=[]

    #Making networks whose number of node is same with the number of member gene in each pathway    
    for GeneSet in GeneSet_Dic.keys():
        Gene_Input=Input(shape=(len(GeneSet_Dic[GeneSet]),),dtype='float32', name=GeneSet+'_Input')
        Drug_effected_Model_for_Attention=[Gene_Input]
        #Drug also affects to the Gene-level network attention mechanism
        Drug_Dense_Geneset=Dense(int(len(GeneSet_Dic[GeneSet])/4)+1,dtype='float32',name=GeneSet+'_Drug')(Drug_Dense_New2)
        Drug_Dense_Geneset=BatchNormalization(name=GeneSet+'_Drug_Batch')(Drug_Dense_Geneset)
        Drug_Dense_Geneset=Activation('relu', name=GeneSet+'Drug_RELU')(Drug_Dense_Geneset)
        Drug_effected_Model_for_Attention.append(Drug_Dense_Geneset) #Drug feature to attention layer

        Gene_Concat=concatenate(Drug_effected_Model_for_Attention,axis=1,name=GeneSet+'_Concat')
        #Gene-level attention network
        Gene_Attention = Dense(len(GeneSet_Dic[GeneSet]), activation='tanh', name=GeneSet+'_Attention_Dense')(Gene_Concat)
        Gene_Attention=Activation(activation='softmax', name=GeneSet+'_Attention_Softmax')(Gene_Attention)
        Attention_Dot=dot([Gene_Input,Gene_Attention],axes=1,name=GeneSet+'_Dot')
        Attention_Dot=BatchNormalization(name=GeneSet+'_BatchNormalized')(Attention_Dot)
        Attention_Dot=Activation('relu',name=GeneSet+'_RELU')(Attention_Dot)

	#Append the list of Gene-level network (attach new pathway)
        GeneSet_Model.append(Attention_Dot)
        GeneSet_Input.append(Gene_Input)

    Drug_effected_Model_for_Attention=GeneSet_Model.copy()

    #Pathway-level network
    Drug_Dense_Sample=Dense(int(len(GeneSet_Dic)/16)+1,dtype='float32',name='Sample_Drug_Dense')(Drug_Dense_New2)
    Drug_Dense_Sample=BatchNormalization(name=GeneSet+'Sample_Drug_Batch')(Drug_Dense_Sample)
    Drug_Dense_Sample=Activation('relu', name='Sample_Drug_ReLU')(Drug_Dense_Sample)    #Drug feature to attention layer
    Drug_effected_Model_for_Attention.append(Drug_Dense_Sample)
    GeneSet_Concat=concatenate(GeneSet_Model,axis=1, name='GeneSet_Concatenate')
    Drug_effected_Concat=concatenate(Drug_effected_Model_for_Attention,axis=1, name='Drug_effected_Concatenate')
    #Pathway-level attention
    Sample_Attention=Dense(len(GeneSet_Dic.keys()),activation='tanh', name='Sample_Attention_Dense')(Drug_effected_Concat)
    Sample_Attention=Activation(activation='softmax', name='Sample_Attention_Softmax')(Sample_Attention)
    Sample_Multiplied=multiply([GeneSet_Concat,Sample_Attention], name='Sample_Attention_Multiplied')
    Sample_Multiplied=BatchNormalization(name='Sample_Attention_BatchNormalized')(Sample_Multiplied)
    Sample_Multiplied=Activation('relu',name='Sample_Attention_Relu')(Sample_Multiplied)
    
    #Making input list
    Input_for_model=[]
    for GeneSet_f in GeneSet_Input:
        Input_for_model.append(GeneSet_f)
    Input_for_model.append(Drug_Input)

    #Concatenate two networks: Pathway-level network, Drug-level network
    Total_model=[Sample_Multiplied,Drug_Dense2]
    Model_Concat=concatenate(Total_model,axis=1, name='Total_Concatenate')

    #Response prediction network
    Concated=Dense(128, name='Total_Dense')(Model_Concat)
    Concated=BatchNormalization(name='Total_BatchNormalized')(Concated)
    Concated=Activation(activation='relu', name='Total_RELU')(Concated)

    Final=Dense(1, name='Output')(Concated)
    Activation(activation='sigmoid', name='Sigmoid')(Final)
    model=Model(inputs=Input_for_model,outputs=Final)

    return model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))



def run(params):
    frm.create_outdir(outdir=params["model_outdir"])
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])

    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")

    # [Req] Set checkpointing
    if params["ckpt_directory"] is None:
        params["ckpt_directory"] = params["model_outdir"]

    # Train
    batch_size = params['batch_size']
    epochs = params['epochs']
    loss = params['loss']
    output_dir = params['output_dir']
    lr = params['learning_rate']

    expr = pd.read_csv(params['ml_data_outdir'] + '/cancer_ge_kegg.csv', index_col=0)
    GeneSet_Dic = json.load(open(params['ml_data_outdir'] + '/geneset.json', 'r'))
    drugs = pd.read_csv(params['ml_data_outdir'] + '/drug_ecfp4_nbits512.csv', index_col=0)

    # Training
    auc_tr = pd.read_csv(params['ml_data_outdir'] + '/train_y_data.csv', index_col=0)
    auc_val = pd.read_csv(params['ml_data_outdir'] + '/val_y_data.csv', index_col=0)
    train_label = auc_tr[params['y_col_name']]
    val_label = auc_val[params['y_col_name']]
    train_input = parse_data(auc_tr, expr, GeneSet_Dic, drugs)
    val_input = parse_data(auc_val, expr, GeneSet_Dic, drugs)

    model_saver = ModelCheckpoint(output_dir + '/model.h5', monitor='val_loss',
                                  save_best_only=True, save_weights_only=False)

    model_stopper = EarlyStopping(monitor='val_loss', restore_best_weights=True,
                                  patience=10)

    callbacks = [model_saver, model_stopper]
    optimizer = Adam(learning_rate=lr)

    model = Making_Model(GeneSet_Dic)
    model.compile(loss=loss, optimizer=optimizer)
    history = model.fit(train_input, train_label, shuffle=True,
                     epochs=epochs, batch_size=batch_size, verbose=2,
                     validation_data=(val_input,val_label),
                     callbacks=callbacks)

    model.save(params["model_outdir"] + '/model.hdf5')
#    model = tf.keras.models.load_model(params["model_outdir"] + '/model.hdf5')
    val_pred = model.predict(val_input).flatten()
    val_label = val_label.to_numpy().flatten()

    frm.store_predictions_df(
        params, y_true=val_label, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"]
    )

    val_scores = frm.compute_performace_scores(
        params, y_true=val_label, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )

    val_loss = np.min(history.history['val_loss'])

    print('IMPROVE_RESULT val_loss:\t' + str(val_loss))

    return val_scores


def main(args):
    additional_definitions = preprocess_params + train_params
    params = frm.initialize_parameters(
        file_path,
        default_model="improve_hidra_default_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    val_scores = run(params)
    print("\nFinished training HiDRA model.")


if __name__ == "__main__":
    main(sys.argv[1:])

    try:
        K.clear_session()

    except AttributeError:
        pass
