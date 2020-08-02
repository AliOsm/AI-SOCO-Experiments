import os
import random
import sklearn
import argparse
import pandas as pd


from simpletransformers.classification import ClassificationModel


from utils import load_labels, load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data_dir')
    parser.add_argument('--models_dir', default='models_dir')
    parser.add_argument('--model_name', default='roberta-small-code-v1')
    args = parser.parse_args()

    train_csv = load_labels(args.data_dir, 'train')
    random.shuffle(train_csv)
    train_labels, train_ids = zip(*train_csv)
    train_labels = list(map(int, train_labels))
    train_ids = list(map(int, train_ids))

    dev_csv = load_labels(args.data_dir, 'dev')
    dev_labels, dev_ids = zip(*dev_csv)
    dev_labels = list(map(int, dev_labels))
    dev_ids = list(map(int, dev_ids))

    train_data = load_data(args.data_dir, 'train', train_csv)
    for i in range(len(train_data)):
    	train_data[i] = train_data[i].replace('    ', '\t')
    train_df = pd.DataFrame(zip(train_data, train_labels))

    dev_data = load_data(args.data_dir, 'dev', dev_csv)
    for i in range(len(dev_data)):
    	dev_data[i] = dev_data[i].replace('    ', '\t')
    dev_df = pd.DataFrame(zip(dev_data, dev_labels))

    model = ClassificationModel(
    	'roberta',
    	os.path.join(args.models_dir, args.model_name),
        num_labels=1000,
    	args={
            'fp16': False,
            'max_seq_length': 512,
            'train_batch_size': 32,
            'eval_batch_size': 32,
            'num_train_epochs': 10,
            'evaluate_during_training': True,
            'evaluate_during_training_steps': True,
            'save_eval_checkpoints': True,
            'logging_steps': 2500,
            'save_steps': 1_000_000_000,
            'save_model_every_epoch': False,
            'overwrite_output_dir': True,
            'sliding_window': True
        }
    )

    model.train_model(train_df, eval_df=dev_df)

    result, model_outputs, wrong_predictions = model.eval_model(dev_df, acc=sklearn.metrics.accuracy_score)
    print(result)
