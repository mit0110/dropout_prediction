import argparse
import os
import re
import pandas
import tensorflow as tf

from kddcup_dataset import KDDCupDataset
from quick_experiment import utils
from models.kdd_lstm import KDDCupLSTMModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_logs_dirname', type=str, default=None,
                        help='Path to directory to store tensorboard info')
    parser.add_argument('--input_directory', type=str,
                        help='The path to the pickled file with the processed'
                             'sequences.')
    parser.add_argument('--test_prediction_dir', type=str,
                        help='The path to the directory to store the '
                             'predictions')
    parser.add_argument('--training_epochs', type=int, default=500,
                        help='The number of epochs to run.')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of times to run the experiment with'
                             'different samples')
    parser.add_argument('--hidden_layer_size', type=int, default=50,
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


def get_periods(args):
    is_valid = lambda filepath: (
        os.path.isfile(filepath) and 'period' in filepath
        and 'c{}_'.format(args.course_number) in filepath)
    for filename in os.listdir(args.input_directory):
        filepath = os.path.join(args.input_directory, filename)
        if not is_valid(filepath):
            continue
        period = int(re.search('.*_period(\d+)_.*', filename).group(1))
        print(filename, period)
        train_df, test_df = utils.pickle_from_file(filepath)
        yield period, train_df, test_df


def evaluate_period(args, experiment_config, kddcup_dataset, period):
    predictions = []
    performances = []
    for run in range(args.runs):
        print('Running iteration {} of {}'.format(run + 1, args.runs))
        kddcup_dataset.set_current_sample(run)
        if args.base_logs_dirname:
            tf.reset_default_graph()
            logs_dirname = os.path.join(
                args.base_logs_dirname,
                'c{}_p{}_run{}'.format(args.course_number, period, run))
            utils.safe_mkdir(logs_dirname)
            experiment_config['logs_dirname'] = logs_dirname
        model = KDDCupLSTMModel(kddcup_dataset, **experiment_config)
        model.fit(partition_name='train',
                  training_epochs=args.training_epochs, close_session=False)

        true, predicted = model.predict('test')
        prediction_df = pandas.DataFrame(true, columns=['true'])
        prediction_df['predicted'] = predicted
        prediction_df['run'] = run
        predictions.append(prediction_df)

        train_performance = pandas.DataFrame(
            model.training_performance, columns=['epoch', 'mse'])
        train_performance.loc[:, 'run'] = run
        train_performance.loc[:, 'dataset'] = 'train'
        val_performance = pandas.DataFrame(
            model.validation_performance, columns=['epoch', 'mse'])
        val_performance.loc[:, 'dataset'] = 'validation'
        val_performance.loc[:, 'run'] = run
        performances.extend([train_performance, val_performance])
    return pandas.concat(predictions), pandas.concat(performances)


def main():
    args = parse_arguments()
    experiment_config, partitions = read_configuration(args)

    print('Dataset Configuration')
    print(partitions)
    print('Experiment Configuration')
    print(experiment_config)

    if args.base_logs_dirname:
        utils.safe_mkdir(args.base_logs_dirname)
    utils.safe_mkdir(args.test_prediction_dir)

    print('Reading dataset')
    predictions, performances = [], []
    for period, train_sequences, test_sequences in get_periods(args):
        print('Creating samples')
        kddcup_dataset = KDDCupDataset(padding_value=-1)
        kddcup_dataset.create_fixed_samples(
            train_sequences.sequence.values, train_sequences.dropout.values,
            test_sequences.sequence.values, test_sequences.dropout.values,
            partition_sizes=partitions, samples_num=args.runs)

        period_predictions, period_performances = evaluate_period(
                args, experiment_config, kddcup_dataset, period)
        period_predictions.loc[:, 'period'] = period
        period_performances.loc[:, 'period'] = period
        performances.append(period_performances)
        predictions.append(period_predictions)

    prediction_filename = os.path.join(
        args.test_prediction_dir,
        'predictions_c{}.p'.format(args.course_number))
    utils.pickle_to_file(pandas.concat(predictions), prediction_filename)

    utils.pickle_to_file(pandas.concat(performances), os.path.join(
        args.test_prediction_dir,
        'performances_c{}.p'.format(args.course_number)))

    print('All operations finished')


if __name__ == '__main__':
    main()
