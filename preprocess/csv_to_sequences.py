"""Script to convert sequences csv into pickle files.

The expected input is the output of the zeppelin notebook ...

The output is a pickled file with a list of numpy arrays (instances) and
a list of labels.
"""

import argparse
import numpy
import os
import pandas
import sys

sys.path.append('./')

from quick_experiment import utils
from sklearn.model_selection import train_test_split
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
    parser.add_argument('--merge', action='store_true',
                        help='Merge the module id with the action type to'
                             'create the vectorizer.')
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
    print('Processing {} sequences'.format(sequences.shape[0]))
    event_encoder = LabelEncoder()
    event_encoder.fit(EVENT_TYPE)

    if not args.merge:
        row_process = lambda x: [x, x.split('-')[1]]
    else:
        row_process = lambda x: x.split('-')
    sequences['sequence'] = sequences['sequence'].apply(
        lambda x: numpy.array([row_process(word) for word in x.split(' ')]))

    # Filter sequences by length
    sequences['len'] = sequences['sequence'].apply(lambda x: x.shape[0])
    sequences = sequences[sequences.len >= args.min_sequence_lenght]

    module_id_encoder = LabelEncoder()
    module_id_encoder.fit(numpy.concatenate(sequences.sequence.values)[:,0])
    sequences['sequence'] = sequences['sequence'].apply(
        lambda xarray: numpy.vstack([module_id_encoder.transform(xarray[:,0]),
                                     event_encoder.transform(xarray[:,1])]).T)

    partitions = train_test_split(
        sequences.sequence.values, sequences.label.values,
        test_size = 0.2, random_state = 42)
    print('Training size: {}. Testing_size: {}'.format(partitions[0].shape[0],
                                                       partitions[1].shape[0]))
    print('Saving sequences')
    utils.pickle_to_file(partitions, args.output_filename)
    print('Saving encoder')
    merged = '-merged' if args.merge else ''
    utils.pickle_to_file(
        module_id_encoder,
        '.'.join(args.output_filename.split('.')[:-1]) + merged + '-encoder.p')
    print('All operations completed')


if __name__ == '__main__':
    main()
