# -*- coding: utf-8 -*-
# +
import pickle
import pandas as pd
from time import time

from tensorflow.keras.models import load_model
from dataset_code.utilities_vec import load_trees_from_array


# -

def show_class_prediction(file, models_path, data_path):
    empirical_vec, empirical_resc_factor = load_trees_from_array([file], return_resc_factor=True)
    n_tips = str(len(empirical_vec[0]))
    classifier = load_model(models_path + 'class/' + n_tips + "_classification_model.keras") 

    start_time = time()
    prediction = classifier.predict(empirical_vec)
    print("--- Classification Inference time:", (time() - start_time), "seconds ---")

    with open(data_path + "raw_" + n_tips + "_10k.pkl", 'rb') as f:
        data = pickle.load(f)
    for i, pred in enumerate(prediction[0]):
        print(data.label_names[i].split('/')[-1].split('_')[0] + f': {pred:.4f}')


reg_values = {
  'BD':['r', 'a'],
  'HE':['r', 'a'],
  'ME':['r', 'a', 'time', 'frac'],
  'SR':['r0', 'r1', 'a0', 'a1', 'time'],
  'WW':['r0', 'r1', 'a0', 'a1', 'time'],
  'SAT':['lambda 0'],
}


def show_reg_prediction(file, div_scenario, models_path, norm=''):
    empirical_vec, empirical_resc_factor = load_trees_from_array([file], return_resc_factor=True)
    n_tips = str(len(empirical_vec[0]))
    regressor = load_model(models_path + 'reg/' + div_scenario + '/' + n_tips + "_regression_" + norm + "model.keras") 
    
    start_time = time()
    y_pred = regressor.predict(empirical_vec)
    print("--- Regression Inference time for EUCALYPTS:", (time() - start_time), "seconds ---")

    data = {'Parameter': reg_values[div_scenario],
            'Predicted value': y_pred[0]}

    return pd.DataFrame(data)
