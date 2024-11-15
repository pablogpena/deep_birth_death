# -*- coding: utf-8 -*-
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from time import time


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
    
    nn_model = load_model(model_path + 'model.keras')
    with open(model_path + 'model_data.pkl', 'rb') as f:
        n_params, train_time = pickle.load(f)
    with open(model_path + 'history.pkl', 'rb') as f:
        history = pickle.load(f)

    # Get classification metrics
    start_time = time()
    y_pred = nn_model.predict(X)
    print("--- Inference time: ", (time() - start_time), "seconds ---")
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
