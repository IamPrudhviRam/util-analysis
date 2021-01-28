import os
import glob
import json
import argparse

from invoicenet import FIELDS
from invoicenet.acp.acp import AttendCopyParse


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--field", nargs='+', type=str, required=True, choices=FIELDS.keys(),
                    help="field to train parser for")
    ap.add_argument("--invoice", type=str, default=None,
                    help="path to directory containing prepared data")
    ap.add_argument("--data_dir", type=str, default='test_data/',
                    help="path to directory containing prepared data")
    ap.add_argument("--pred_dir", type=str, default='predictions/',
                    help="path to directory containing prepared data")

    args = ap.parse_args()

    paths = []
    fields = []
    predictions = {}

    if args.invoice:
        if not os.path.exists(args.invoice):
            print("Could not find file '{}'".format(args.invoice))
            return
        paths.append(args.invoice)
    else:
        paths = [os.path.abspath(f) for f in glob.glob(args.data_dir + "**/*.pdf", recursive=True)]

    if not os.path.exists('./models/invoicenet/'):
        print("Could not find any trained models!")
        return
    else:
        models = os.listdir('./models/invoicenet/')
        for field in args.field:
            if field in models:
                fields.append(field)
            else:
                print("Could not find a trained model for field '{}', skipping...".format(field))

    for field in fields:
        print("\nExtracting field '{}' from {} invoices...\n".format(field, len(paths)))
        model = AttendCopyParse(field=field, restore=True)
        predictions[field] = model.predict(paths=paths)

    os.makedirs(args.pred_dir, exist_ok=True)
    for idx, filename in enumerate(paths):
        filename = os.path.basename(filename)[:-3] + 'json'
        labels = {}
        if os.path.exists(os.path.join(args.pred_dir, filename)):
            with open(os.path.join(args.pred_dir, filename), 'r') as fp:
                labels = json.load(fp)
        with open(os.path.join(args.pred_dir, filename), 'w') as fp:
            print("\nFilename: {}".format(filename))
            for field in predictions.keys():
                labels[field] = predictions[field][idx]
                print("  {}: {}".format(field, labels[field]))
            fp.write(json.dumps(labels, indent=2))
            print('\n')

    print("Predictions stored in '{}'".format(args.pred_dir))


if __name__ == '__main__':
    main()