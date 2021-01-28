import argparse

from invoicenet.common import trainer
from invoicenet.parsing.parser import Parser
from invoicenet.parsing.data import ParseData
from invoicenet.acp.data import InvoiceData


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--field", type=str, required=True, choices=["amount", "date"],
                    help="field to train parser for")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="batch size for training")
    ap.add_argument("--restore", action="store_true",
                    help="restore from checkpoint")
    ap.add_argument("--steps", type=int, default=50000,
                    help="maximum number of training steps")
    ap.add_argument("--early_stop_steps", type=int, default=0,
                    help="stop training if validation doesn't improve "
                         "for a given number of steps, disabled when 0 (default)")

    args = ap.parse_args()

    output_length = {"date": InvoiceData.seq_date, "amount": InvoiceData.seq_amount}[args.field]

    train_data = ParseData.create_dataset(
        path='invoicenet/parsing/data/%s/train.tsv' % args.field,
        output_length=output_length,
        batch_size=args.batch_size)

    val_data = ParseData.create_dataset(
        path='invoicenet/parsing/data/%s/valid.tsv' % args.field,
        output_length=output_length,
        batch_size=args.batch_size)

    print("Training...")
    trainer.train(
        model=Parser(field=args.field, restore=args.restore),
        train_data=train_data,
        val_data=val_data,
        total_steps=args.steps,
        early_stop_steps=args.early_stop_steps
    )


if __name__ == '__main__':
    main()
