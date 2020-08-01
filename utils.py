import os
import csv


from tqdm import tqdm


def load_labels(data_dir, split):
    with open(os.path.join(data_dir, '{}.csv'.format(split)), 'r') as fp:
        reader = csv.reader(fp)
        problems = list(reader)
    problems = problems[1:]
    return problems


def load_data(data_dir, split, split_csv):
    problems = list()
    for row in tqdm(split_csv):
        with open(os.path.join(data_dir, split, row[1]), 'r') as fp:
            problems.append(fp.read())
    return problems
