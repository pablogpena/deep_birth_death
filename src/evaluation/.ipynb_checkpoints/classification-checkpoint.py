# -*- coding: utf-8 -*-
# +
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from time import time
from tensorflow.keras.models import Model


# -

# # Results structure
# * 674
#  * accuracy
#  * f1
#  * recall
#  * precision
#  
#  * n_params
#  * train_time
#  * history
#  * y_pred
#  * y_test
# * 489
#  * ...
#
# * 87
#  * ...

# # Generate results

def generate_class_results(model_path, X, y):
    results = dict()
    
    nn_model = load_model(model_path + 'temperature_model.keras')
    with open(model_path + 'temperature_model_data.pkl', 'rb') as f:
        n_params, train_time = pickle.load(f)
    with open(model_path + 'temperature_history.pkl', 'rb') as f:
        history = pickle.load(f)

    # Get classification metrics
    start_time = time()
    y_pred = nn_model.predict(X)
    print("--- Testing time normal model: ", (time() - start_time), "seconds ---")
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y, axis=1)

    report = classification_report(y_test, y_pred, output_dict=True)

    # Save all results
    results['accuracy'] = report['accuracy']
    results['f1-score'] = report['macro avg']['f1-score']
    results['recall'] = report['macro avg']['recall']
    results['precision'] = report['macro avg']['precision']

    results['n_params'] = n_params
    results['train_time'] = train_time
    results['history'] = history
    results["y_pred"] = y_pred
    results["y_test"] = y_test
    
    return results


# # Calibrate the model

# +
def fit_TemperatureCalibration(train_X_y, valid_X_y=None, epochs=100):
    
    ### inspired by: https://github.com/stellargraph/stellargraph/blob/develop/stellargraph/calibration.py ###
    
    T = tf.Variable(tf.ones(shape=(1,)))
    history = []
    early_stopping = False
    optimizer = Adam(learning_rate=0.001)
    
    def cost(T, x, y):

        scaled_logits = tf.multiply(x=x, y=1.0 / T)

        cost_value = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=scaled_logits, labels=y)
        )

        return cost_value

    def grad(T, x, y):

        with tf.GradientTape() as tape:
            cost_value = cost(T, x, y)

        return cost_value, tape.gradient(cost_value, T)
    
    
    X_train, y_train = train_X_y
    if valid_X_y:
        X_valid, y_valid = valid_X_y
        early_stopping = True
    
    
    for epoch in range(epochs):
        train_cost, grads = grad(T, X_train, y_train)
        optimizer.apply_gradients(zip([grads], [T]))
        if early_stopping:
            val_cost = cost(T, X_valid, y_valid)
            if (len(history) > 0) and (val_cost > history[-1][1]):
                break
            else: 
                history.append([train_cost, val_cost, T.numpy()[0]])
        else:
            history.append([train_cost, T.numpy()[0]])

    history = np.asarray(history)
    temperature = history[-1, -1]
    
    return temperature


def calibrated_proba(logits, temperature):
    
    scaled_prediction = logits / temperature

    return np.exp(scaled_prediction) / np.sum(np.exp(scaled_prediction), axis=-1, keepdims=True)


# -
def generate_class_results_calibrated_model(model_path, X, y):
    results_calibrated_model = dict()

    split_idx = int(len(X) * 0.5)
    x_train, x_test = X[split_idx:], X[:split_idx]
    y_train, y_test = y[split_idx:], y[:split_idx]
    
    
    nn_model = load_model(model_path + 'temperature_model.keras')
        
    with open(model_path + 'temperature_model_data.pkl', 'rb') as f:
        n_params, train_time = pickle.load(f)
    with open(model_path + 'temperature_history.pkl', 'rb') as f:
        history = pickle.load(f)
    
    #Calibrate the model.
    start_time = time()
    model_score = Model(inputs=nn_model.input, outputs=nn_model.get_layer('logits').output)
    
    X_train_calib = model_score.predict(x_train)
    X_valid_calib = model_score.predict(x_test)
    
    temperature = fit_TemperatureCalibration((X_train_calib,y_train), (X_valid_calib,y_test), epochs=1000)

    pred_logit = model_score.predict(X)
    pred_calib = calibrated_proba(pred_logit, temperature)
    print("--- Testing time temperature model: ", (time() - start_time), "seconds ---")
    y_pred = np.argmax(pred_calib, axis=1)
    
    y_test = np.argmax(y, axis=1)
    report = classification_report(y_test, y_pred, output_dict=True)
    
     # Save all results
        
    
    results_calibrated_model['accuracy'] = report['accuracy']
    results_calibrated_model['f1-score'] = report['macro avg']['f1-score']
    results_calibrated_model['recall'] = report['macro avg']['recall']
    results_calibrated_model['precision'] = report['macro avg']['precision']

    results_calibrated_model['n_params'] = n_params
    results_calibrated_model['train_time'] = train_time
    results_calibrated_model['history'] = history
    results_calibrated_model["y_pred"] = y_pred
    results_calibrated_model["y_test"] = y_test
    results_calibrated_model["temperature"] = temperature
    
        
    
    return results_calibrated_model


# # Classification metrics

def get_classification_results(results):
    columns_list = []
    for i in results:
        columns_list.append(i)
    df = pd.DataFrame(columns=columns_list)

    for i in results:
        df.at['accuracy', i] = (results[i]['accuracy'])
        df.at['F1-Score', i] = (results[i]['f1-score'])
        df.at['Recall', i] = (results[i]['recall'])
        df.at['Precision', i] = (results[i]['precision'])
        df.at['Number of params', i] = (results[i]['n_params'])
        df.at['Train time', i] = (results[i]['train_time'])

    return df


def get_MLE_classification_results(results):
    columns_list = []
    for i in results:
        columns_list.append(i)
    df = pd.DataFrame(columns=columns_list)

    for i in results:
        df.at['accuracy', i] = (results[i]['accuracy'])
        df.at['F1-Score', i] = (results[i]['f1-score'])
        df.at['Recall', i] = (results[i]['recall'])
        df.at['Precision', i] = (results[i]['precision'])

    return df


# # Confusion matrix

def plot_conf_mat(y_pred, y_val, names, i):
    f, ax = plt.subplots(1, figsize=(5, 5))

    sns.heatmap(confusion_matrix(y_val, y_pred), annot=True,
                cmap='Blues', fmt = 'd', ax=ax, cbar=False)

    ax.set_ylabel("Real label")
    ax.set_xlabel("Predicted label")
    
    names = [os.path.basename(path) for path in names]
    
    ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    ax.xaxis.set_ticklabels(names)
    ax.set_yticklabels(ax.get_yticks(), rotation = 45)
    ax.yaxis.set_ticklabels(names)
    ax.set_title(i + ' tips')

    plt.show()

    print(classification_report(y_val, y_pred, digits=4))


# # Train plots 

def train_plot(ax, i, metric, results):
    ax.plot(results[i]['history'][metric],
            linestyle='-', label='Train', color='blue', linewidth=.7)
    ax.plot(results[i]['history']['val_' + metric],
            linestyle='--', label='Validation', color='blue', linewidth=.7)
    

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_title(i)
