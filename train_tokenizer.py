import os
import argparse


from tokenizers import ByteLevelBPETokenizer


from utils import load_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data_dir')
    args = parser.parse_args()

    train_csv = load_labels(args.data_dir, 'train')

    files = list()
    for row in train_csv:
        files.append(os.path.join(args.data_dir, 'train', row[1]))

    tokenizer = ByteLevelBPETokenizer(lowercase=False)
    tokenizer.train(files, vocab_size=32000, min_frequency=1, special_tokens=[
        '<s>',
        '<pad>',
        '</s>',
        '<unk>',
        '<mask>',
    ])

    tokenizer.save('./tokenizer', 'tokenizer-model')
