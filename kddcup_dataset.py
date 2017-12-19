import numpy

from quick_experiment.dataset import SequenceDataset


class KDDCupDataset(SequenceDataset):
    """Class to abstract a KDDCup log dataset.

    Each instance is a sequence of ids, labeled with a binary value.

    The KDDCup dataset keeps the test split fixed over samples and only
    changes the train and validation splits.
    """
    def create_fixed_samples(self, train_instances, train_labels,
                             test_instances, test_labels, samples_num,
                             partition_sizes):
        """Creates samples with a random partition generator.

        Args:
            train_instances (:obj: iterable): instances to divide in samples.
            test_instances (:obj: iterable): instances to divide in samples.
            train_labels (:obj: iterable): labels to divide in samples.
            test_labels (:obj: iterable): labels to divide in samples.
            samples_num (int): the number of samples to create.
            partition_sizes (dict): a map from the partition names to their
                proportional sizes. The sum of train and validation must be
                less or equal to one, and the value of test must be less or
                equal to one.
        """
        assert 'train' in partition_sizes and 'validation' in partition_sizes
        assert partition_sizes['train'] + partition_sizes['validation'] <= 1.0
        assert partition_sizes['test'] <= 1.0
        assert train_labels is not None and test_labels is not None
        assert train_instances.shape[0] == train_labels.shape[0]
        assert test_instances.shape[0] == test_labels.shape[0]

        self.samples_num = samples_num
        self._sample_indices = [
            dict.fromkeys(partition_sizes) for _ in range(samples_num)]
        self._instances = numpy.hstack([train_instances, test_instances])
        self._labels = numpy.hstack([train_labels, test_labels])
        self._test_start = train_instances.shape[0]

        for sample in range(self.samples_num):
            self._sample_indices[sample] = self._split_sample(partition_sizes)

    @property
    def maximums(self):
        """Returns the maximum value for a one hot encoding per instance column.
        """
        if not hasattr(self, '_maximums'):
            self._maximums = numpy.max(
                [instance.max(axis=0) for instance in self._instances], axis=0)
        return self._maximums

    def _split_sample(self, partition_sizes):
        sample_index = {}
        if 'test' in partition_sizes:
            test_indices = numpy.arange(self._test_start,
                                        self._instances.shape[0])
            numpy.random.shuffle(test_indices)
            total_instances = partition_sizes['test'] * test_indices.shape[0]
            sample_index['test'] = test_indices[:int(total_instances)]

        indices = numpy.arange(self._test_start)
        numpy.random.shuffle(indices)
        partitions = [(name, proportion)
                      for name, proportion in partition_sizes.items()
                      if name != 'test']
        cumulative_sizes = numpy.cumsum([x[1] for x in partitions])
        splits = numpy.split(
            indices, (cumulative_sizes * indices.shape[0]).astype(numpy.int32))
        # The last split is the remaining portion.
        for partition, split in zip(partitions, splits[:-1]):
            sample_index[partition[0]] = split

        return sample_index