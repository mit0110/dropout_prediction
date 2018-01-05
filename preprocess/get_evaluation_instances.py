"""Script to extract evaluation instances for recommendation.

The evaluation instances are taken from the test sequences.

Each instance is an group of ngrams that:
  * share the same prefix. The size of the prefix varies from 2 to n-1.
  * occurs in more than min_freq sequences

The output of the script, if output_filename is provided, is a pickled dict
mapping from each prefix to the suffixes and instances where they occur.

{
    prefix [tuple]: {
        suffix [tuple]: (
            instances [list of numpy_arrays]
            labels [list of ints]
        )
    }
}
"""
import numpy
import argparse
import sys

sys.path.append('./')

from collections import defaultdict
from quick_experiment import utils


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', type=str,
                        help='The path to the pickled file with the processed'
                             'sequences.')
    parser.add_argument('--output_filename', type=str, default=None,
                        help='The path to the file to store the evaluation '
                             'instances')
    parser.add_argument('--explore', action='store_true',
                        help='Explore different combinations and print obtained'
                             'sizes.')
    parser.add_argument('--N', type=int, default=3,
                        help='The size of the ngrams.')
    parser.add_argument('--min_freq', type=int, default=5,
                        help='The minimum frequency for an instance.')

    return parser.parse_args()


def get_ngram_positions(N, sequences):
    ngram_positions = defaultdict(dict)
    for index, sequence in enumerate(sequences):
        for i in range(len(sequence) - N + 1):
            ngram = tuple(sequence[i: i + N][:, 0].tolist())
            ngram_positions[ngram][index] = i
    return ngram_positions


def get_evaluation_ngrams(MIN_FREQ, ngram_positions, suffix_size=1):
    evaluation_ngrams = defaultdict(list)
    for ngram, indices in ngram_positions.items():
        if len(indices) < MIN_FREQ:
            continue
        evaluation_ngrams[ngram[:-suffix_size]].append(ngram[-suffix_size:])
    filtered_evaluation = {prefix: suffixes
                           for prefix, suffixes in evaluation_ngrams.items()
                           if len(suffixes) >= 2}
    return filtered_evaluation


def get_instances(filtered_ngrams, labels, ngram_positions, sequences):
    evaluation_instances = defaultdict(dict)
    for prefix, sufixes in filtered_ngrams.items():
        for sufix in sufixes:
            ngram = prefix + sufix
            # https://stackoverflow.com/questions/835092/
            # python-dictionary-are-keys-and-values-always-the-same-order
            lengths = [x for x in ngram_positions[ngram].values()]
            instances = []
            instances_labels = []
            for index, length in ngram_positions[ngram].items():
                instance = sequences[index][:length + len(prefix)]
                assert numpy.array_equal(instance[-len(prefix):, 0], prefix)
                instances.append(instance)
                instances_labels.append(index)
            evaluation_instances[prefix][sufix] = (instances, instances_labels)
            assert len(instances) == len(instances_labels)
    return evaluation_instances


def get_suffixes(N, labels, min_freq, sequences):
    sufixes_dict = {}
    for suffix_size in range(1, N - 1):
        ngram_positions = get_ngram_positions(N, sequences)
        filtered_ngrams = get_evaluation_ngrams(
            min_freq, ngram_positions, suffix_size)

        evaluation_instances = get_instances(filtered_ngrams, labels,
                                             ngram_positions, sequences)
        print('min_freq {} / N {} / suffix_size {}'.format(
            min_freq, N, suffix_size))
        print('Prefixes found: {}'.format(len(evaluation_instances)))
        print('Total instances found: {}'.format(sum(
            [len(instances[0]))
             for suffixes_dict in evaluation_instances.values()
             for instances in suffixes_dict.values()]
        )))
        sufixes_dict[suffix_size] = evaluation_instances
    return sufixes_dict


def main():
    args = parse_arguments()
    raw_sequences = utils.pickle_from_file(args.input_filename)
    sequences = raw_sequences[1]
    labels = raw_sequences[3]

    if args.explore:
        for min_freq in [3, 5, 10]:
            for N in range(3, 6):
                get_suffixes(
                    N, labels, min_freq, sequences)
    else:
        evaluation_instances = get_suffixes(
            args.N, labels, args.min_freq, sequences)
        if args.output_filename is not None:
            utils.pickle_to_file(evaluation_instances,
                                 args.output_filename)
    print('All operations completed')


if __name__ == '__main__':
    main()

