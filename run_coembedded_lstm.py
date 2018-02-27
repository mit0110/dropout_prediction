import argparse
import numpy
import os
import tensorflow as tf

from gensim.models import Word2Vec
from kddcup_dataset import KDDCupDataset
from quick_experiment import utils
from models import kdd_coembedded_lstm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_logs_dirname', type=str, default=None,
                        help='Path to directory to store tensorboard info')
    parser.add_argument('--filename', type=str,
                        help='The path to the pickled file with the processed'
                             'sequences.')
    parser.add_argument('--test_prediction_dir', type=str,
                        help='The path to the directory to store the '
                             'predictions')
    parser.add_argument('--training_epochs', type=int, default=1000,
                        help='The number of epochs to run.')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of times to run the experiment with'
                             'different samples')
    parser.add_argument('--hidden_layer_size', type=int, default=100,
                        help='Number of cells in the recurrent layer.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number if instances to process at the same time.')
    parser.add_argument('--log_values', type=int, default=50,
                        help='How many training epochs to wait before logging'
                             'the accuracy in validation.')
    parser.add_argument('--max_num_steps', type=int, default=100,
                        help='Number of time steps to unroll the network.')
    parser.add_argument('--dropout_ratio', type=float, default=0.3,
                        help='Dropout for the input layer and the recurrent '
                             'layer.')
    parser.add_argument('--course_number', type=str,
                        help='Number of the course to identify predictions.')
    parser.add_argument('--embedding_model', type=str, default=None,
                        help='Path to word2vec model to use as pretrained '
                             'embeddings.')
    parser.add_argument('--nofinetune', action='store_true',
                        help='Do no change the pretrained embedding.')
    parser.add_argument('--log_gradients', action='store_true',
                        help='Log gradients and learning rate.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--model', type=str, default='abs',
                        help='Name of the model to run. The variation is in the'
                             'difference function between co-embeddings. '
                             'Possible values are abs, square, biabs and '
                             'bisquare.')

    return parser.parse_args()


MODELS = {
    'abs': kdd_coembedded_lstm.KDDCupCoEmbeddedLSTMModel,
    'square': kdd_coembedded_lstm.KDDCupCoEmbeddedLSTMModel2,
    'norm': kdd_coembedded_lstm.KDDCupCoEmbeddedLSTMModel3,
    'norm-fixed': kdd_coembedded_lstm.KDDCupCoEmbeddedLSTMModel4,
    'biabs': kdd_coembedded_lstm.KDDCupCoEmbedBiLSTMModel,
    'bisquare': kdd_coembedded_lstm.KDDCupCoEmbedBiLSTMModel2
}


def read_configuration(args):
    config = {
        'hidden_layer_size': args.hidden_layer_size,
        'batch_size': args.batch_size,
        'log_values': args.log_values,
        'max_num_steps': args.max_num_steps,
        'dropout_ratio': args.dropout_ratio,
        'finetune_embeddings': not args.nofinetune,
        'name': args.model,
        'log_gradients': args.log_gradients,
        'learning_rate': args.learning_rate,
    }
    dataset_config = {'train': 0.85, 'test': 1, 'validation': 0.15}
    return config, dataset_config


def transform_input(train_sequences, test_sequences):
    """Removes the second column from all sequences. Leaves only the module id.
    """
    return (
        numpy.array([x[:,0] for x in train_sequences]),
        numpy.array([x[:,0] for x in test_sequences])
    )


def read_embedding_model(model_path):
    if model_path is None:
        return None
    return Word2Vec.load(model_path)


def main():
    args = parse_arguments()
    experiment_config, partitions = read_configuration(args)

    print('Reading dataset')
    train_sequences, test_sequences, train_labels, test_labels =\
        utils.pickle_from_file(args.filename)
    train_sequences, test_sequences = transform_input(train_sequences,
                                                      test_sequences)
    print('Creating samples')
    embedding_model = read_embedding_model(args.embedding_model)
    kddcup_dataset = KDDCupDataset(embedding_model=embedding_model)
    kddcup_dataset.create_fixed_samples(
        train_sequences, train_labels, test_sequences, test_labels,
        partition_sizes=partitions, samples_num=args.runs)

    kddcup_dataset.set_current_sample(0)

    print('Dataset Configuration')
    print(partitions)
    print('Experiment Configuration')
    print(experiment_config)
    print('Pretrained embedding model')
    print(args.embedding_model)

    if args.base_logs_dirname:
        utils.safe_mkdir(args.base_logs_dirname)
    utils.safe_mkdir(args.test_prediction_dir)

    for run in range(args.runs):
        print('Running iteration {} of {}'.format(run + 1, args.runs))
        kddcup_dataset.set_current_sample(run)
        if args.base_logs_dirname:
            tf.reset_default_graph()
            logs_dirname = os.path.join(
                args.base_logs_dirname,
                'c{}_run{}'.format(args.course_number, run))
            utils.safe_mkdir(logs_dirname)
            experiment_config['logs_dirname'] = logs_dirname

        model = MODELS[args.model](
            kddcup_dataset, embedding_model=embedding_model,
            **experiment_config)
        model.fit(partition_name='train',
                  training_epochs=args.training_epochs, close_session=False)

        predicted_labels = model.predict('test')
        prediction_dirname = os.path.join(
            args.test_prediction_dir,
            'predictions_c{}_run{}.p'.format(args.course_number, run))
        utils.pickle_to_file(predicted_labels, prediction_dirname)

        utils.pickle_to_file(
            (model.training_performance, model.validation_performance),
            os.path.join(
                args.test_prediction_dir,
                'performances_c{}_run{}.p'.format(args.course_number, run)))

    print('All operations finished')


if __name__ == '__main__':
    main()
