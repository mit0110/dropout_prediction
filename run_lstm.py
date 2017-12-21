import argparse
import os
import tensorflow as tf

from kddcup_dataset import KDDCupDataset
from quick_experiment import utils
from models.kdd_lstm import KDDCupLSTMModel


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

    return parser.parse_args()


def read_configuration(args):
    config = {
        'hidden_layer_size': args.hidden_layer_size,
        'batch_size': args.batch_size,
        'log_values': args.log_values,
        'max_num_steps': args.max_num_steps,
        'dropout_ratio': args.dropout_ratio
    }
    dataset_config = {'train': 0.85, 'test': 1, 'validation': 0.15}
    return config, dataset_config


def main():
    args = parse_arguments()
    experiment_config, partitions = read_configuration(args)

    print('Reading dataset')
    train_sequences, test_sequences, train_labels, test_labels =\
        utils.pickle_from_file(args.filename)
    print('Creating samples')
    kddcup_dataset = KDDCupDataset(padding_value=-1)
    kddcup_dataset.create_fixed_samples(
        train_sequences, train_labels, test_sequences, test_labels,
        partition_sizes=partitions, samples_num=args.runs)

    kddcup_dataset.set_current_sample(0)

    print('Dataset Configuration')
    print(partitions)
    print('Experiment Configuration')
    print(experiment_config)

    if args.base_logs_dirname:
        utils.safe_mkdir(args.base_logs_dirname)
    utils.safe_mkdir(args.test_prediction_dir)

    for run in range(args.runs):
        print('Running iteration {} of {}'.format(run + 1, args.runs))
        kddcup_dataset.set_current_sample(run)
        if args.base_logs_dirname:
            tf.reset_default_graph()
            logs_dirname = os.path.join(args.base_logs_dirname,
                                        'run{}'.format(run))
            utils.safe_mkdir(logs_dirname)
            experiment_config['logs_dirname'] = logs_dirname
        model = KDDCupLSTMModel(kddcup_dataset, **experiment_config)
        model.fit(partition_name='train',
                  training_epochs=args.training_epochs,
                  close_session=False)

        predicted_labels = model.predict('test')
        prediction_dirname = os.path.join(
            args.test_prediction_dir, 'predictions_run{}.p'.format(run))
        utils.pickle_to_file(predicted_labels, prediction_dirname)

        utils.pickle_to_file(
            (model.training_performance, model.validation_performance),
            os.path.join(args.test_prediction_dir,
                         'performances_run{}.p'.format(run)))

    print('All operations finished')


if __name__ == '__main__':
    main()
