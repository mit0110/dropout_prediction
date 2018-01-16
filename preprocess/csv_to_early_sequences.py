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
    parser.add_argument('--sequences_dirname', type=str,
                        help='The path to the directory with the sequences csv '
                             'files.')
    parser.add_argument('--enrollments_dirname', type=str,
                        help='The path to the directory with the enrollments '
                             'csv files.')
    parser.add_argument('--labels_filename', type=str,
                        help='The path to the csv file with the labels.')
    parser.add_argument('--output_filename', type=str,
                        help='The path to store the pickled sequences.')
    parser.add_argument('--min_sequence_lenght', type=int, default=1,
                        help='Only include sequences with lenght grater than'
                             'this.')
    parser.add_argument('--merge', action='store_true',
                        help='Merge the module id with the action type to'
                             'create the vectorizer.')
    parser.add_argument('--period_span', type=int, default=4,
                        help='Amount of days of a training period')
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


def read_logs(args):
    sequences_files = get_input_filenames(args.sequences_dirname,
                                          extension='csv')
    df_from_each_file = (pandas.read_csv(f, header=0, parse_dates=['time'],
                                         infer_datetime_format=True)
                         for f in sequences_files)
    sequences = pandas.concat(df_from_each_file, ignore_index=True)

    enrollment_files = get_input_filenames(args.enrollments_dirname, '.csv')
    df_from_each_file = (pandas.read_csv(f, header=0,
                                         usecols=['course_id', 'enrollment_id'])
                         for f in enrollment_files)
    courses_df = pandas.concat(df_from_each_file, ignore_index=True)

    labels_df = pandas.read_csv(args.labels_filename,
                             names=['enrollment_id', 'dropout'])
    sequences = sequences.merge(courses_df, on='enrollment_id', how='left')
    sequences = sequences.merge(labels_df, on='enrollment_id', how='left')

    return sequences


def get_sequences(logs_df, period_span):
    min_time = logs_df['time'].min()
    logs_df['time_lapsed'] = logs_df['time'] - min_time
    logs_df['days_lapsed'] = logs_df['time_lapsed'].apply(
        lambda date: date.days)
    max_day = logs_df['days_lapsed'].max()
    for period_end in range(period_span, max_day, period_span):
        period_logs = logs_df[logs_df.days_lapsed <= period_end + period_span]
        train_enrollments, test_enrollments = train_test_split(
            period_logs['enrollment_id'])
        train_period = period_logs[period_logs.days_lapsed <= period_end]
        test_period = period_logs[period_logs.days_lapsed > period_end]
        labels = period_logs[['enrollment_id', 'days_lapsed']].groupby(
            'enrollment_id').max().rename(columns={'days_lapsed':'last_day'})






def main():
    args = parse_arguments()
    logs_df = read_logs(args)
    event_encoder = LabelEncoder()
    event_encoder.fit(EVENT_TYPE)
    logs_df['event'] = event_encoder.transform(logs_df['event'])

    for course_id in logs_df.course_id.unique():
        course_logs = logs_df[logs_df.course_id == course_id]
        course_sequences = get_sequences(course_logs, args.period_span)
    print('Processing {} sequences'.format(sequences.shape[0]))
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
        test_size=0.2, random_state=42)
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
