import tensorflow as tf
import math
import numpy as np
import pandas as pd
import os

# You can use these methods below to consider class imbalance problems.
# In this article, we used BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, SVMSMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN

import kerastuner as kt

# Allocating GPU Memory
GPU_NUM=0
selected_gpu = tf.config.list_physical_devices('GPU')[GPU_NUM]
tf.config.experimental.set_visible_devices(selected_gpu, 'GPU')
tf.config.experimental.set_memory_growth(selected_gpu, True)

def make_dir(fol_name):
    if not os.path.exists(fol_name):
        os.makedirs(fol_name)


for i in range(10, 105, 5):
    print(str(i)+'th start!!')
    dir_name = "20220319_9_r"+str(i) 

    make_dir("model/"+dir_name)

    train = pd.read_csv("data/train.csv")
    val = pd.read_csv("data/val.csv")
    
    train_Y = train.pop('PEB')
    train_X = train.copy(deep=True)

    val_Y = val.pop('PEB')
    val_X = val.copy(deep=True)
    
    over_sam1 = BorderlineSMOTE(sampling_strategy=i/100, random_state=1)
    train_X2, train_Y2 = over_sam1.fit_sample(train_X, train_Y)
    
    train_X2.to_csv("model/"+dir_name+"/tr_X_BorderlineSMOTE"+str(i)+".csv")
    train_Y2.to_csv("model/"+dir_name+"/tr_Y_BorderlineSMOTE"+str(i)+".csv")

    # vars - normalization
    train_X.Age = (train_X.Age - train.Age.mean())/(train.Age.std())
    train_X.Size_tumor = (train_X.Size_tumor - train.Size_tumor.mean())/(train.Size_tumor.std())
    train_X.Albumin_preESD = (train_X.Albumin_preESD - train.Albumin_preESD.mean())/(train.Albumin_preESD.std())
    train_X.INR_preESD = (train_X.INR_preESD - train.INR_preESD.mean())/(train.INR_preESD.std())

    train_X2.Age = (train_X2.Age - train.Age.mean())/(train.Age.std())
    train_X2.Size_tumor = (train_X2.Size_tumor - train.Size_tumor.mean())/(train.Size_tumor.std())
    train_X2.Albumin_preESD = (train_X2.Albumin_preESD - train.Albumin_preESD.mean())/(train.Albumin_preESD.std())
    train_X2.INR_preESD = (train_X2.INR_preESD - train.INR_preESD.mean())/(train.INR_preESD.std())

    val_X.Age = (val_X.Age - train.Age.mean())/(train.Age.std())
    val_X.Size_tumor = (val_X.Size_tumor - train.Size_tumor.mean())/(train.Size_tumor.std())
    val_X.Albumin_preESD = (val_X.Albumin_preESD - train.Albumin_preESD.mean())/(train.Albumin_preESD.std())
    val_X.INR_preESD = (val_X.INR_preESD - train.INR_preESD.mean())/(train.INR_preESD.std())

    neg, pos = np.bincount(train_Y2)
    total = neg+pos

    train_Y_bce = tf.keras.utils.to_categorical(train_Y)
    train_Y2_bce = tf.keras.utils.to_categorical(train_Y2)
    val_Y_bce = tf.keras.utils.to_categorical(val_Y)

    initial_bias = np.log([pos/neg])

    def build_model(hp):
        inputs = tf.keras.Input(shape=(len(train_X.columns),1))
        hp_units = hp.Int('units', min_value = 12, max_value = 24, step = 4)
        qx = tf.keras.layers.Dense(units=hp_units, kernel_initializer='glorot_uniform', activation='selu')(inputs)
        x = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=2)(qx, qx)
        x = tf.keras.layers.Add()([x, qx])
        layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x) # Add+Norm
        x = tf.keras.layers.Dense(hp_units, activation='selu')(layernorm1) # point-wise feed-forward
        x = tf.keras.layers.Dense(hp_units)(x)
        x = tf.keras.layers.Add()([layernorm1, x])
        x = tf.keras.layers.Flatten()(x)
        hp_units2 = hp.Int('units2', min_value = 12, max_value = 24, step = 4)
        x_bce = tf.keras.layers.Dense(units=hp_units2, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer='l1_l2')(x)
        outputs_bce = tf.keras.layers.Dense(units=1, activation='sigmoid', name='outputs_bce')(x_bce)

        model = tf.keras.Model(inputs=inputs, outputs=[outputs_bce])

        hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=hp_learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0),
                      metrics=tf.keras.metrics.AUC(name='auc', curve='ROC')
                      #metrics=tf.keras.metrics.Recall(name='recall')
                     )
        return model


    tuner = kt.BayesianOptimization(build_model,
                         objective = kt.Objective("val_loss", direction="min"),
                         max_trials = 20,                    
                         directory = '../model/'+dir_name,
                         project_name = 'tuning')

    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg)*(total)/2
    weight_for_1 = (1 / pos)*(total)/2

    class_weight = {0: weight_for_0, 1: weight_for_1}

    # Searching and training the models
    tuner.search(train_X2, train_Y2, epochs = 20, validation_split=0.2)#, class_weight=class_weight)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    units = best_hps.get('units')
    units2 = best_hps.get('units2')
    lr = best_hps.get('learning_rate')

    model = tuner.get_best_models(num_models=1)[0]

    # Save the network and prediction result
    results = model.predict(val_X)
    pd.DataFrame(data=results, columns=['PROB']).to_csv("../model/"+dir_name+"/pred_prob.csv")
    
    model.save("../model/"+dir_name+"/model", save_format="tf")
    model.save_weights("../model/"+dir_name+"/model_weights")

    pred_val_Y = model.predict(val_X)
    pred_train_Y = model.predict(train_X)

    print(roc_auc_score(train_Y.values, pred_train_Y),
          roc_auc_score(val_Y.values, pred_val_Y))

    res_df = pd.DataFrame({'DATASET':['BorderlineSMOTE'+str(i)],
                           'UNITS':[str(units)], 'UNITS2':[str(units2)], 'OPTIMIZER':['ADAM'], 'LR':[str(lr)],
                           'METRICS':['AUC_PR'], 
                           'TR_AUC':[str(roc_auc_score(train_Y.values, pred_train_Y))], 
                           'VAL_AUC':[str(roc_auc_score(val_Y.values, pred_val_Y))]})


    res_df.to_csv("../model/"+dir_name+"/result.csv")

