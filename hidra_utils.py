import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate, multiply, dot, Activation
from tensorflow.keras.utils import Sequence


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
    genes = GeneSet.intersection(df.columns[1:])
    cols = [canc_col_name] + sorted(list(genes))

    return df[cols], GeneSet_Dic


def parse_data(ic50, expr, GeneSet_Dic, drugs):
    # Divide expression data based on pathway
    X = []

    for pathway in GeneSet_Dic.keys():
        df = expr[GeneSet_Dic[pathway]]
        X.append(df.loc[ic50['improve_sample_id']])

    X.append(drugs.loc[ic50['improve_chem_id']])

    return X


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def Making_Model(GeneSet_Dic):
    # Original HiDRA model with keras
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


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, expr, genes, labels, indexes, batch_size, y_name, drugs=False):
        'Initialization'
        self.labels = labels
        self.expr = expr
        self.genes = genes
        self.batch_size = batch_size
        self.y_col = y_name
        self.indexes = indexes
        self.is_drugs = drugs

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.labels.shape[0] / self.batch_size))

    def __getitem__(self):
        'Generate one batch of data'
        batch_labels = self.labels.loc[self.indexes]

        # Generate data
        X = self.__data_generation(batch_labels)
        y = batch_labels[self.y_col]

        return X, y

    def __data_generation(self, batch_labels):
        'Divide expression data based on pathway, for batch_size samples'

        if self.is_drugs:
            df = self.expr
            X = df.loc[batch_labels['improve_chem_id']]

        else:
            df = self.expr[self.genes]
            X = df.loc[batch_labels['improve_sample_id']]

        return X


class MultiGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, expr, drugs, GeneSet_Dic, labels, batch_size, y_name, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.y_col = y_name
        self.expr = expr
        self.drugs = drugs
        self.geneset_dic = GeneSet_Dic
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.labels.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_labels = self.labels.loc[indexes]

        # Generate data
        X = self.__data_generation(batch_labels, indexes)
        y = np.asarray([x for x in batch_labels[self.y_col]])

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.asarray([x for x in self.labels.index])

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_labels, indexes):
        'Divide expression data based on pathway, for batch_size samples'
        X = []

        for pathway in self.geneset_dic.keys():
            genes = self.geneset_dic[pathway]
            d_g = DataGenerator(self.expr, genes, self.labels, indexes, self.batch_size, self.y_col)
            X_gen, y = d_g.__getitem__()
            X.append(X_gen)

        d_g = DataGenerator(self.drugs, genes, self.labels, indexes, self.batch_size, self.y_col, drugs=True)
        X_gen, y = d_g.__getitem__()
        X.append(X_gen)

        return X
