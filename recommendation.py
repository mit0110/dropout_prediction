import argparse
import numpy
import os
import tensorflow as tf

from gensim.models import Word2Vec
from itertools import combinations_with_replacement
from kddcup_dataset import KDDCupDataset
from quick_experiment import utils
from models.kdd_lstm import KDDCupLSTMModel
from models.kdd_embedded_lstm import KDDCupEmbeddedLSTMModel
from models.kdd_coembedded_lstm import KDDCupCoEmbeddedLSTMModel
from tqdm import tqdm


MODELS = {
    'lstm': KDDCupLSTMModel,
    'embedded': KDDCupEmbeddedLSTMModel,
    'coembedded': KDDCupCoEmbeddedLSTMModel
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_logs_dirname', type=str, default=None,
                        help='Path to directory with the tensorboard model')
    parser.add_argument('--filename', type=str,
                        help='The path to the pickled file with the processed'
                             'evaluation sequences.')
    parser.add_argument('--output_filename', type=str,
                        help='The path to the directory to store the '
                             'evaluated rankings')
    parser.add_argument('--hidden_layer_size', type=int, default=100,
                        help='Number of cells in the recurrent layer.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number if instances to process at the same time.')
    parser.add_argument('--max_num_steps', type=int, default=100,
                        help='Number of time steps to unroll the network.')
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='Number of units in the embedding layer.')
    parser.add_argument('--embedding_model', type=str, default=None,
                        help='Path to word2vec model to use as pretrained '
                             'embeddings.')
    parser.add_argument('--model_type', type=str, default='coembedded',
                        help='The type of model to use')
    # parser.add_argument('--encoder', type=str,
    #                     help='The path to the encoder file to get the largest '
    #                           'possible suffix.')

    return parser.parse_args()


def read_embedding_model(model_path):
    if model_path is None:
        return None
    return Word2Vec.load(model_path)


def read_configuration(args):
    config = {
        'hidden_layer_size': args.hidden_layer_size,
        'batch_size': args.batch_size,
        'max_num_steps': args.max_num_steps,
    }
    if args.model_type == 'embedded':
        config['embedding_size'] = args.embedding_size

    if args.model_type != 'lstm':
        config['embedding_model'] = read_embedding_model(args.embedding_model)
    return config


def transform_input(sequences):
    """Removes the second column from all sequences. Leaves only the module id.
    """
    return numpy.array([x[:,0] for x in sequences])


def transform_labels(labels):
    return numpy.sign(numpy.array(labels) - 0.5)


def possible_suffixes(suffix_size, max_id):
    return combinations_with_replacement(range(max_id), suffix_size)


def get_instances_for_suffix(instances, suffix, possible_suffixes, labels):
    """For each instance, concatenate all possible suffixes.

    To distinguish between real and non-present labels, we change the label
    0 to -1 and add label 0 where the combination instance-suffix is
    artificial.
    """
    eval_instances = []
    eval_labels = []
    for instance, label in zip(transform_input(instances),
                               transform_labels(labels)):
        for possible_suffix in possible_suffixes:
            eval_instances.append(numpy.concatenate([instance, possible_suffix]))
            if possible_suffix == suffix:
                eval_labels.append(label)
            else:
                eval_labels.append(0)  # Neutral label
    return eval_instances, eval_labels


def evaluate_predictions(true, predictions, batch_size):
    assert true.shape[0] % batch_size == 0
    positive_rankings = []
    negative_rankings = []
    for i in range(0, true.shape[0], batch_size):
        batch_true = true[i: i+batch_size]
        assert numpy.count_nonzero(batch_true) == 1
        batch_predicted = predictions[i: i+batch_size]
        true_instance_index = numpy.where(batch_true != 0)[0][0]
        ranking = len(numpy.where(batch_predicted >=
                                  batch_predicted[true_instance_index])[0])
        if batch_true[true_instance_index] < 0:
            negative_rankings.append((ranking, batch_size))
        else:
            positive_rankings.append((ranking, batch_size))
    return positive_rankings, negative_rankings


def evaluate_prefix(prefix, suffix_dict, model, possible_suffixes):
    results = []
    eval_instances = []
    eval_labels = []
    for suffix, (instances, labels) in suffix_dict.items():
        # Create the dataset
        eval_suf_instances, eval_suf_labels = get_instances_for_suffix(
            instances, suffix, possible_suffixes, labels)
        eval_instances.extend(eval_suf_instances)
        eval_labels.extend(eval_suf_labels)
    print(len(eval_instances))
    dataset = KDDCupDataset(embedding_model=model.embedding_model)
    dataset.create_fixed_samples(
        numpy.array([]), numpy.array([], dtype=numpy.int32),
        numpy.array(eval_instances),
        numpy.array(eval_labels, dtype=numpy.int32),
        partition_sizes={'train': 1,'validation': 0, 'test': 1},
        samples_num=1)
    dataset.set_current_sample(0)
    model.dataset = dataset
    if model.graph is None:
        model.build_all()

    # Obtain prediction. It is very important the dataset does not change
    # the order of the test partition
    true, predictions = model.predict(partition_name='test')
    # Now we know the first len(possible_suffixes) predictions correspond
    # to the first instance concatenated with all possible suffixes.
    results.append(evaluate_predictions(
        true, predictions, len(possible_suffixes)))
    return results


def main():
    args = parse_arguments()
    experiment_config = read_configuration(args)

    print('Reading dataset')
    possible_suffixes, evaluation_instances = utils.pickle_from_file(
        args.filename)
    # encoder = utils.pickle_from_file(args.encoder)

    print('Experiment Configuration')
    print(experiment_config)
    print('Pretrained embedding model')
    print(args.embedding_model)
    model = MODELS[args.model_type](None, **experiment_config)
    results = []
    for suffix_length, prefix_dict in evaluation_instances.items():
        print('Evaluating suffixes with length {}'.format(suffix_length))
        for prefix, suffix_dict in tqdm(prefix_dict.items()):
            results.extend(evaluate_prefix(prefix, suffix_dict, model,
                                           possible_suffixes[suffix_length]))

    print('Saving results')
    utils.pickle_to_file(results, args.output_filename)
    print('All operations completed')


if __name__ == '__main__':
    main()
