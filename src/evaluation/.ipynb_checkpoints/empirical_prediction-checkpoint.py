# -*- coding: utf-8 -*-
# +
import pickle
import pandas as pd
from time import time
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from dataset_code.utilities_vec import load_trees_from_array


# -

def calibrated_proba(logits, temperature):
    
    scaled_prediction = logits / temperature

    return np.exp(scaled_prediction) / np.sum(np.exp(scaled_prediction), axis=-1, keepdims=True)


# +
def show_class_prediction(file, models_path, data_path, temperature_path):
    empirical_vec, empirical_resc_factor = load_trees_from_array([file], return_resc_factor=True)
    n_tips = str(len(empirical_vec[0]))

    classifier = load_model(models_path + 'class/' + n_tips + "_classification_model.keras") 
    
    temperature_classifier = load_model(models_path + 'class/' + n_tips + "_classification_temperaturemodel.keras")
    model_score = Model(inputs=temperature_classifier.input, outputs=temperature_classifier.get_layer('logits').output)    
    
    with open(temperature_path + "temperature_" + n_tips + ".txt", "r") as file:
        temperature = file.read()
    temperature = float(temperature.strip())    
    
    
    
    start_time = time()
    
    prediction = classifier.predict(empirical_vec)
    

    with open(data_path + "raw_" + n_tips + "_10k.pkl", 'rb') as f:
        data = pickle.load(f)
        
    print("----- No temperature model prediction -----")
    
    for i, pred in enumerate(prediction[0]):
        print(data.label_names[i].split('/')[-1].split('_')[0] + f': {pred:.4f}')
        
   

    ## Temparature model ###
    
    

        
        
    pred_logit = model_score.predict(empirical_vec)
    pred_calib = calibrated_proba(pred_logit, temperature)
    
    
    print("----- Temperature model prediction -----")
    for i, pred in enumerate(pred_calib[0]):
        print(data.label_names[i].split('/')[-1].split('_')[0] + f': {pred:.4f}')        
        
        
    ### Temperature + mcmc dropout 
    
    num_samples = 10000
    preds = []
    for _ in range(num_samples):
        p = model_score(empirical_vec, training=True)  
        pred_calib = calibrated_proba(p, temperature)
        
        preds.append(pred_calib)  
            
    preds = np.array(preds)# shape: (num_samples, test_size, 4) 
    predictions = np.mean(preds, axis=0)
    
    print("----- Temperature model + MCMC Dropout prediction -----")   
    for i, pred in enumerate(predictions[0]):
        print(data.label_names[i].split('/')[-1].split('_')[0] + f': {pred:.4f}')
    
        
    print("--- Classification Inference time:", (time() - start_time), "seconds ---")    
        
            
# -

reg_values = {
  'BD':['r', 'a'],
  'HE':['r', 'a'],
  'ME':['r', 'a', 'time', 'frac'],
  'SR':['r0', 'r1', 'a0', 'a1', 'time'],
  'WW':['r0', 'r1', 'a0', 'a1', 'time'],
  'SAT':['lambda 0'],
}


# +
#Posterior with Monte Carlo dropout
#import numpy as np
#
#num_samples = 1000
#preds = []
#for _ in range(num_samples):
#    # Note: We pass training=True to keep dropout active
#    p = estimator([X_dyn_test[2:3],X_rep_test[2:3]], training=True)  # dropout is active here
#    preds.append(p.numpy())  # if using eager execution
#
#preds = np.array(preds)  # shape: (num_samples, test_size, 4)
#mean_pred = preds.mean(axis=0)      # (test_size, 4)
#std_pred = preds.std(axis=0)        # (test_size, 4)
# -

def show_reg_prediction(file, div_scenario, models_path, norm=''):
    empirical_vec, empirical_resc_factor = load_trees_from_array([file], return_resc_factor=True)
    print("AAAAAAAAAAAAAAAAAAAAAA")
    print(empirical_vec.shape)
    n_tips = str(len(empirical_vec[0]))
    regressor = load_model(models_path + 'reg/' + div_scenario + '/' + n_tips + "_regression_" + norm + "model.keras") 
    
    start_time = time()
    
    num_samples = 1000
    
    y_pred = regressor.predict(empirical_vec)
    
    
    ### MCMC dropout to get CI ###       
    
    preds = []
    for _ in range(num_samples):
        # Note: We pass training=True to keep dropout active
        p = regressor(empirical_vec, training=True)  # dropout is active here
        preds.append(p.numpy())  # if using eager execution
    
    preds = np.array(preds)# shape: (num_samples, test_size, 4)
    
    #Rescale back
    
    if div_scenario == "BD" or div_scenario == "HE" or div_scenario == "DD":
        preds[:, :, 0] /= empirical_resc_factor[0]
        y_pred[:, 0] /= empirical_resc_factor[0]
    
    if div_scenario == "ME":
        preds[:, :, 0] /= empirical_resc_factor[0]
        preds[:, :, 2] /= empirical_resc_factor[0]
        
        y_pred[:, 0] /= empirical_resc_factor[0]
        y_pred[:, 2] /= empirical_resc_factor[0]
  
    else:
        preds[:, :, 0] /= empirical_resc_factor[0]
        preds[:, :, 1] /= empirical_resc_factor[0]
        preds[:, :, 4] /= empirical_resc_factor[0]
        
        y_pred[:, 0] /= empirical_resc_factor[0]
        y_pred[:, 1] /= empirical_resc_factor[0]
        y_pred[:, 4] /= empirical_resc_factor[0]
        
        

    mean_pred = preds.mean(axis=0)     
    std_pred = preds.std(axis=0) 
    data = {'Parameter': reg_values[div_scenario],
            'Predicted value': y_pred[0]}


    print("--- Regression Inference time", (time() - start_time), "seconds ---")


    return pd.DataFrame(data), preds, mean_pred, std_pred
