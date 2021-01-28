import sys
import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import pandas as pd


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    compare_invoices()


def compare_invoices():
    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir is not '' else '.'
    labelled_data_dir_path = current_dir + '/labelled'
    predicted_data_dir_path = current_dir + '/predictions'
    labelled_file_names = os.listdir(labelled_data_dir_path)
    predicted_file_names = os.listdir(predicted_data_dir_path)
    print("labelled file names", labelled_file_names)
    labels = [
        'customer_name', 'bill_date', 'start_date', 'end_date', 'total_amount'
    ]
    # print("label(0)", labels)
    labelled_dict = {}
    predicted_dict = {}
    for file in labelled_file_names:
        # print("path", labelled_data_dir_path + "/" + file)
        with open(labelled_data_dir_path + "/" + file, encoding='utf-8') as f:
            file_dict = json.loads(f.read())
            # print("file dict", file_dict)
            for key in file_dict:
                # print("labelled key values", key, file_dict[key])
                temp_key = key.replace(key, str(labels.index(key)))
                # temp_dict[temp_key] = file+" "+file_dict[key]
                labelled_dict[file+key+" "+file_dict[key]] = temp_key
            # print("file dict", file_dict)
    print("labelled Dict is: ", labelled_dict)

    for file in predicted_file_names:
        # print("path", predicted_data_dir_path + "/" + file)
        with open(predicted_data_dir_path + "/" + file, encoding='utf-8') as f:
            file_dict = json.loads(f.read())
            # print("file dict", file_dict)
            for key in file_dict:
                # print("labelled key values", key, file_dict[key])
                temp_key = key.replace(key, str(labels.index(key)))
                # temp_dict[temp_key] = file+" "+file_dict[key]
                predicted_dict[file+key+" "+file_dict[key]] = temp_key
            # print("file dict", file_dict)
    print("Predicte Dict is: ", predicted_dict)

    common_items = {k: predicted_dict[k] for k in predicted_dict if k in labelled_dict}
    combined_items = {k: labelled_dict[k] for k in labelled_dict}
    diff_items = {k: predicted_dict[k] for k in predicted_dict if
                  k in labelled_dict and predicted_dict[k] != labelled_dict[k]}
    matched_items = {k: predicted_dict[k] for k in predicted_dict if
                  k in labelled_dict and predicted_dict[k] == labelled_dict[k]}
    print("length matched comm", len(common_items))
    print("length matched comb", len(combined_items))
    print("length matched diff", len(diff_items))
    print("length matched matc", len(matched_items))

    y_true = []
    y_predict = []
    # for key in common_items:
    #     # print("labelled key values", key, labelled_dict[key])
    #     y_true.append(labelled_dict[key])
    #     # print("predicte key values", key, predicted_dict[key])
    #     y_predict.append(predicted_dict[key])
    # print("Done\n")

    for key in combined_items.keys():
        y_true.append(labelled_dict[key])
        if key in predicted_dict.keys():
            y_predict.append(predicted_dict[key])
        else:
            y_predict.append('5')
    print("Done\n")

    print("yTrue values", y_true)
    print("yPred values", y_predict)
    confusion_matrix(y_true, y_predict)
    print("confusion matrix value : \n ", confusion_matrix(y_true, y_predict, labels=np.unique(y_predict)))
    print("accuracy score value : \t ", accuracy_score(y_true, y_predict))
    print("Classification report  : \n ", classification_report(y_true, y_predict))
    print("recall score value :  ", recall_score(y_true, y_predict, average=None, labels=np.unique(y_predict)))
    print("precision score value :  ", precision_score(y_true, y_predict, average=None, labels=np.unique(y_predict)))


if __name__ == '__main__':
    main()
