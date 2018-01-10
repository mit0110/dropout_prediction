import numpy

from quick_experiment.dataset import SequenceDataset


class KDDCupDataset(SequenceDataset):
    """Class to abstract a KDDCup log dataset.

    Each instance is a sequence of ids, labeled with a binary value.

    The KDDCup dataset keeps the test split fixed over samples and only
    changes the train and validation splits.

    Args:
        padding_value: (int) value to pad the sequences when they are shorter
            than max_sequence_length.
        embedding_model: (gensim.models.Word2Vec) A model trained with the
            same sequences to use in instances. It will be used to transform the
            vocabulary of the sequences into the correct indices of the
            embeddings.
            The new instances will be sequences with the index of the word in
            the embedding matrix plus one. This is to accomodate the 0 embedding
            for the padding elements. If the element is not in the embedding
            the assigned index in len(embeddings).
            As a result, the obtained indexes work with an embedding with TWO
            more columns, one at the beginning for padding elements and one at
            the end for infrequent elements
    """

    def __init__(self, padding_value=0, embedding_model=None):
        super(KDDCupDataset, self).__init__(padding_value=padding_value)
        self._maximums = None
        self.embedding_model = embedding_model

    def classes_num(self, _=None):
        return 2

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
        if self.embedding_model is not None:
            self._fit_embedding_vocabulary()
        self._labels = numpy.hstack([train_labels, test_labels])
        self._test_start = train_instances.shape[0]

        for sample in range(self.samples_num):
            self._sample_indices[sample] = self._split_sample(partition_sizes)

    @property
    def maximums(self):
        """Returns the maximum value for a one hot encoding per instance column.
        """
        if self._maximums is None:
            self._maximums = numpy.max(
                [instance.max(axis=0) for instance in self._instances], axis=0)
        return self._maximums

    def _split_sample(self, partition_sizes):
        sample_index = {}
        if 'test' in partition_sizes:
            test_indices = numpy.arange(self._test_start,
                                        self._instances.shape[0])
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

    def _fit_embedding_vocabulary(self):
        assert self.padding_value == 0
        word2index = {
            word: index
            for index, word in enumerate(self.embedding_model.wv.index2word)
        }
        # We have to add one to the result because the 0 embedding is for the
        # padded element of the sequence.
        map_function = numpy.vectorize(
            lambda x: word2index.get(str(x), len(word2index)) + 1)

        self._instances = numpy.array([
            map_function(sequence) for sequence in self._instances
        ])
