import os
import argparse


from tokenizers import ByteLevelBPETokenizer


from utils import load_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data_dir')
    args = parser.parse_args()

    tokenizer = ByteLevelBPETokenizer(lowercase=False)
    tokenizer.train([os.path.join(args.data_dir, 'train.cat')], vocab_size=32000, min_frequency=1, special_tokens=[
        '<s>',
        '<pad>',
        '</s>',
        '<unk>',
        '<mask>',
    ])

    tokenizer.save('./tokenizer', 'roberta-code')
