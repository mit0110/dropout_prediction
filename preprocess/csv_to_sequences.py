"""Script to convert sequences csv into pickle files.

The expected input is the output of the zeppelin notebook ...

The output is a pickled file with a list of numpy arrays (instances) and
a list of labels.
"""

import argparse
import numpy
import os
import pandas
import re
import sys

sys.path.append('./')

from quick_experiment import utils
from sklearn.preprocessing import LabelEncoder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirname', type=str,
                        help='The path to the directory with the csv files.')
    parser.add_argument('--output_filename', type=str,
                        help='The path to store the pickled sequences.')
    parser.add_argument('--min_sequence_lenght', type=int, default=1,
                        help='Only include sequences with lenght grater than'
                             'this.')
    return parser.parse_args()


def get_input_filenames(input_dirpath, extension):
    """Returns the names of the files in input_dirpath that matches pattern."""
    all_files = os.listdir(input_dirpath)
    result = []
    for filename in all_files:
        if filename.endswith(extension) and os.path.isfile(os.path.join(
                input_dirpath, filename)):
            result.append(os.path.join(input_dirpath, filename))
    return result


EVENT_TYPE = ["problem", "video", "access", "wiki", "discussion",
              "navigate", "page_close"]

def main():
    args = parse_arguments()
    input_filenames = get_input_filenames(args.input_dirname,
                                          extension='csv')
    print('Reading {} files'.format(len(input_filenames)))
    df_from_each_file = (pandas.read_csv(f, header=0)
                         for f in input_filenames)
    sequences = pandas.concat(df_from_each_file, ignore_index=True)
    print('Processing sequences')
    encoder = LabelEncoder()
    encoder.fit(EVENT_TYPE)
    row_process = lambda x1, x2: [int(float(x1)), encoder.transform([x2])[0]]
    sequences['sequence'] = sequences['sequence'].apply(
            lambda x: numpy.array([row_process(*x.split('-'))
                                   for x in x.split(' ')], dtype=numpy.int32))
    sequences['len'] = sequences['sequence'].apply(lambda x: x.shape[0])
    sequences = sequences[sequences.len >= args.min_sequence_lenght]
    print('Saving sequences')
    utils.pickle_to_file((sequences.sequence.values, sequences.label.values),
                         args.output_filename)
    print('All operations completed')


if __name__ == '__main__':
    main()
