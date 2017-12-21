"""Script to train word2vec models over sequences."""

import argparse
import os
import numpy
import pandas

from gensim.models import Word2Vec


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirname', type=str,
                        help='The path to the directory with the csv files.')
    parser.add_argument('--output_filename', type=str,
                        help='The path to store the obtained model.')
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='The size of the output embedding.')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of times to iterate over the dataset.')
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


def main():
    args = parse_arguments()
    input_filenames = get_input_filenames(args.input_dirname,
                                          extension='csv')
    print('Reading {} files'.format(len(input_filenames)))
    df_from_each_file = (pandas.read_csv(f, header=0)
                         for f in input_filenames)
    sequences = pandas.concat(df_from_each_file, ignore_index=True)
    print('Processing {} sequences'.format(sequences.shape[0]))

    sequences['sequence'] = sequences['sequence'].apply(
        lambda xarray: numpy.squeeze(xarray[:, 0]))  # Only leave first column

    model_config = {
        "size": args.embedding_size,
        "window": 5,
        "workers": 6,
        "min_count": 5,
        "alpha": 0.01,  # initial learning rate
        "iter": args.iterations,
        "negative": 5,
        "sg": 1,  # Use skipgram model
        # "hs": 1 if args.hs else 0,
        # "cbow_mean": 1 if args.cbow_mean else 0
    }

    print('Start fitting with config {}'.format(model_config))
    model = Word2Vec(sequences.sequence.values, **model_config)
    print('Saving model')
    model.save(args.output_filename)
    print('All iterations completed')


if __name__ == '__main__':
    main()