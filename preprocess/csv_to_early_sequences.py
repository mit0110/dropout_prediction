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
    parser.add_argument('--logs_dirname', type=str,
                        help='Path to the directory with the log csv files.')
    parser.add_argument('--enrollments_dirname', type=str,
                        help='Path to the directory with the enrollments '
                             'csv files.')
    parser.add_argument('--labels_filename', type=str,
                        help='Path to the csv file with the labels.')
    parser.add_argument('--output_directory', type=str,
                        help='Path to directory to store the output sequences.')
    parser.add_argument('--min_sequence_lenght', type=int, default=0,
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
    print('Reading logs')
    sequences_files = get_input_filenames(args.logs_dirname,
                                          extension='csv')
    df_from_each_file = (pandas.read_csv(f, header=0, parse_dates=['time'],
                                         infer_datetime_format=True)
                         for f in sequences_files)
    logs_df = pandas.concat(df_from_each_file, ignore_index=True)

    enrollment_files = get_input_filenames(args.enrollments_dirname, '.csv')
    df_from_each_file = (pandas.read_csv(f, header=0,
                                         usecols=['course_id', 'enrollment_id'])
                         for f in enrollment_files)
    courses_df = pandas.concat(df_from_each_file, ignore_index=True)

    labels_df = pandas.read_csv(args.labels_filename,
                             names=['enrollment_id', 'dropout'])
    logs_df = logs_df.merge(courses_df, on='enrollment_id', how='left')
    logs_df = logs_df.merge(labels_df, on='enrollment_id', how='left')
    print('Reading logs: finished')
    return logs_df


def merge_objects(logs_df, merge=False):
    module_id_encoder = LabelEncoder()
    if merge:
        logs_df.loc[:,'object'] = logs_df[['object', 'event']].apply(
            lambda row: '-'.join([str(x) for x in row]), axis=1)
    logs_df.loc[:,'object'] = module_id_encoder.fit_transform(logs_df['object'])
    logs_df['object_pair'] = [x for x in zip(logs_df.object, logs_df.event)]
    return logs_df, module_id_encoder


def collect_sequences(logs_df):
    return logs_df.groupby(
        'enrollment_id')['object_pair'].apply(list)


def get_sequences(logs_df, period_span):
    min_time = logs_df['time'].min()
    logs_df.loc[:,'time_lapsed'] = logs_df['time'] - min_time
    logs_df.loc[:,'days_lapsed'] = logs_df['time_lapsed'].apply(
        lambda date: date.days)
    max_day = logs_df['days_lapsed'].max()
    period = 0
    for period, period_end in enumerate(range(
            period_span, max_day - period_span, period_span)):
        period_logs = logs_df[logs_df.days_lapsed <= period_end + period_span]
        labels = period_logs[['enrollment_id', 'days_lapsed']].groupby(
            'enrollment_id').max().rename(columns={'days_lapsed':'last_day'})
        labels = (labels.last_day <= period_end).astype(int).to_frame()
        train_period = period_logs[period_logs.days_lapsed <= period_end]
        sequences = collect_sequences(train_period).to_frame()
        yield period, sequences.merge(labels, left_index=True, right_index=True,
                                      how='left')

    complete_sequences = collect_sequences(logs_df).to_frame()
    labels = logs_df[['enrollment_id', 'dropout']].drop_duplicates(
        subset='enrollment_id', keep='last').set_index('enrollment_id')
    yield period + 1, complete_sequences.merge(
        labels, left_index=True, right_index=True, how='left')


def main():
    args = parse_arguments()
    logs_df = read_logs(args)
    event_encoder = LabelEncoder()
    event_encoder.fit(EVENT_TYPE)
    logs_df.loc[:,'event'] = event_encoder.transform(logs_df['event'])

    # Filter sequences by length
    if args.min_sequence_lenght > 0:
        lens = logs_df.groupby('enrollment_id')[
            'dropout'].count().rename(columns={'dropout': 'sequence_len'})
        valid_enrollments = lens[lens > 5].index.values
        print('Removing {} sequences'.format(
            lens.shape[0] - valid_enrollments.shape[0]))
        logs_df = logs_df[logs_df.enrollment_id.isin(valid_enrollments)]

    for course_id in logs_df.course_id.unique():
        print('Processing course {}'.format(course_id))
        course_logs = logs_df[logs_df.course_id == course_id]
        course_logs, encoder = merge_objects(course_logs, args.merge)
        train_enrollments, test_enrollments = train_test_split(
            course_logs['enrollment_id'].unique())
        for period, sequences in get_sequences(course_logs, args.period_span):
            train_sequences = sequences.loc[sequences.index.intersection(
                train_enrollments).tolist()]
            test_sequences = sequences.loc[sequences.index.intersection(
                test_enrollments).tolist()]
            print('Period{}, train {}, test {}'.format(
                period, train_sequences.shape[0], test_sequences.shape[0]))

            filename = 'c{}_span{}_period{}{}.p'.format(
                course_id, args.period_span, period,
                '_merged' if args.merge else '')
            utils.pickle_to_file(
                (train_sequences, test_sequences),
                os.path.join(args.output_directory, filename))

        utils.pickle_to_file(encoder, os.path.join(
            args.output_directory, 'c{}_span{}_encoder{}.p'.format(
                course_id, args.period_span, '_merged' if args.merge else '')))
    print('All operations completed')


if __name__ == '__main__':
    main()
