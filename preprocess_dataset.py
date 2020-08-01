import os
import random
import argparse


from utils import load_labels, load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data_dir')
    args = parser.parse_args()

    train_csv = load_labels(args.data_dir, 'train')
    train_labels, train_ids = zip(*train_csv)
    train_labels = list(map(int, train_labels))
    train_ids = list(map(int, train_ids))
    random.shuffle(train_csv)

    dev_csv = load_labels(args.data_dir, 'dev')
    dev_labels, dev_ids = zip(*dev_csv)
    dev_labels = list(map(int, dev_labels))
    dev_ids = list(map(int, dev_ids))

    train_data = load_data(args.data_dir, 'train', train_csv)
    for i in range(len(train_data)):
    	train_data[i] = train_data[i].replace('    ', '\t')

    dev_data = load_data(args.data_dir, 'dev', dev_csv)
    for i in range(len(dev_data)):
    	dev_data[i] = dev_data[i].replace('    ', '\t')

    with open(os.path.join(args.data_dir, 'train.cat'), 'w') as fp:
    	for example in train_data:
    		fp.write(example + '\n')

    with open(os.path.join(args.data_dir, 'dev.cat'), 'w') as fp:
    	for example in dev_data:
    		if example[-1] == '\n':
    			fp.write(example)
    		else:
    			fp.write(example + '\n')
