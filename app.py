from flask import Flask
from pyngrok import ngrok
from flasgger import Swagger
from flask import request

import os
import glob
import json
import dateparser
import datetime
from invoicenet.acp.acp import AttendCopyParse

app = Flask(__name__)
Swagger(app)
public_url = ngrok.connect(5000).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, 5000))


@app.route('/', methods=["Get"])
def hello():

    return '<h2>Invoice Api working</h2>'


@app.route('/main', methods=['POST'])
def main():
    """Let's Predict Invoice bills
        This is using docstrings for specifications.
        ---
        consumes:
          - multipart/form-data
        parameters:
          - name: files
            in: formData
            type: file
            required: true
        responses:
            500:
                description: ERROR Failed!
            200:
                description: Success!
    """

    # # Postman-FormData
    print("formz", request.files)
    # _field = request.form['field']
    _file = request.files.getlist('files')
    # _invoice = request.form['invoice']
    # _data_dir = request.form['data_dir']
    # _pred_dir = request.form['predicted_dir']

    _field = '{"key1": "invoice_number", "key2": "vendor_name", "key3": "invoice_date", "key4": "tax_amount","key5": "total_amount"}'

    dummy = []
    for index, item in enumerate(_file):
        dummy.append(item)
        _file[index] = item.read()
    print("dummy", dummy)
    print("dummy", len(_file))

    _invoice = ''
    _data_dir = 'test_data/'
    _pred_dir = 'predictions/'

    # Postman-args
    # _field = request.args.get('field')
    # _invoice = request.args.get('invoice')
    # _data_dir = request.args.get('data_dir')
    # _pred_dir = request.args.get('predicted_dir')

    if _data_dir == '':
        _data_dir = 'test_data/'
    if _pred_dir == '':
        _pred_dir = 'predictions/'

    ## File path as input
    # paths = []
    # fields = []
    # predictions = {}
    #
    # if _invoice:
    #     if not os.path.exists(_invoice):
    #         print("Could not find file '{}'".format(_invoice))
    #         return
    #     paths.append(_invoice)
    # else:
    #     paths = [os.path.abspath(f) for f in glob.glob(_data_dir + "**/*.pdf", recursive=True)]
    #
    # if not os.path.exists('./models/invoicenet/'):
    #     print("Could not find any trained models!")
    #     return
    # else:
    #     models = os.listdir('./models/invoicenet/')
    #     # print("type", type(_field))
    #     _field = json.loads(_field)
    #     # print("type 1", type(_field), _field)
    #     # print("value", _field['key1'])
    #     for key in _field:
    #         # print("field 1", _field[key])
    #         if _field[key] in models:
    #             value = _field[key]
    #             fields.append(value)
    #         else:
    #             print("Could not find a trained model for field '{}', skipping...".format(_field[key]))
    # # print("fieldss", type(fields), fields)
    # for field in fields:
    #     print("\nExtracting field '{}' from {} invoices...\n".format(field, len(paths)))
    #     model = AttendCopyParse(field=field, restore=True)
    #     predictions[field] = model.predict(paths=paths)
    #
    # os.makedirs(_pred_dir, exist_ok=True)
    # for idx, filename in enumerate(paths):
    #     filename = os.path.basename(filename)[:-3] + 'json'
    #     labels = {}
    #     if os.path.exists(os.path.join(_pred_dir, filename)):
    #         with open(os.path.join(_pred_dir, filename), 'r') as fp:
    #             labels = json.load(fp)
    #     with open(os.path.join(_pred_dir, filename), 'w') as fp:
    #         print("\nFilename: {}".format(filename))
    #         # print("\nPredictions: {}".format(predictions))
    #         for field in predictions.keys():
    #             # print("pred 1", predictions[field][idx])
    #             labels[field] = predictions[field][idx]
    #             print("  {}: {}".format(field, labels[field]))
    #         fp.write(json.dumps(labels, indent=2))
    #         print('\n')
    # print("field", _field)

    ## Byte Stream as input
    paths = []
    fields = []
    predictions = {}

    if _invoice:
        if not os.path.exists(_invoice):
            print("Could not find file '{}'".format(_invoice))
            return
        paths.append(_invoice)
    else:
        # paths = [os.path.abspath(f) for f in glob.glob(_data_dir + "**/*.pdf", recursive=True)]
        paths = _file
        # for item in paths:
        #     print("pathz type", type(item))
        # print('pathz', paths)
        # print('pathz', type(_file))
        # paths = _file

    if not os.path.exists('./models/invoicenet/'):
        print("Could not find any trained models!")
        return
    else:
        models = os.listdir('./models/invoicenet/')
        _field = json.loads(_field)
        for key in _field:
            if _field[key] in models:
                value = _field[key]
                fields.append(value)
            else:
                print("Could not find a trained model for field '{}', skipping...".format(_field[key]))
    for field in fields:
        print("\nExtracting field '{}' from {} invoices...\n".format(field, len(paths)))
        model = AttendCopyParse(field=field, restore=True)
        predictions[field] = model.predict(paths=paths)

    os.makedirs(_pred_dir, exist_ok=True)
    for idx, filename in enumerate(paths):
        filename = os.path.basename(str(idx))[:-3] + '.json'
        labels = {}
        if os.path.exists(os.path.join(_pred_dir, filename)):
            with open(os.path.join(_pred_dir, filename), 'r') as fp:
                labels = json.load(fp)
        with open(os.path.join(_pred_dir, filename), 'w') as fp:
            print("\nFilename: {}".format(filename))
            print("\nPredictions: {}".format(predictions))
            for field in predictions.keys():
                print("pred 1", predictions)
                labels[field] = predictions[field][idx]
                print("  {}: {}".format(field, labels[field]))
            fp.write(json.dumps(labels, indent=2))
            print('\n')
            print("labelz", labels)
    print("field", _field)
    print("file type", type(_file))
    print("labelz", labels)

    # print("date", predictions['invoice_date'])
    dummy = [dateparser.parse(date, settings={'DATE_ORDER': 'MDY', 'PREFER_LOCALE_DATE_ORDER': False}) for date in predictions['invoice_date']]
    to_string = [dateTime.strftime('%d-%m-%Y') for dateTime in dummy]
    # print("date 1", predictions['invoice_date'], to_string)
    predictions['invoice_date'] = to_string
    # print("final date", predictions['invoice_date'])
    print("Predictions stored in '{}'".format(_pred_dir))
    print("predictions", predictions)
    # return '<h2>Utility bills predicted, Check Predicted folder for results</h2><p>Predictions : \n{prediction}</p>'.format(prediction=predictions)
    return '{prediction}'.format(prediction=predictions)
