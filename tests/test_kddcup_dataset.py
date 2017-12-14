import numpy
import random
import unittest

from kddcup_dataset import KDDCupDataset


class KDDCupDatasetTest(unittest.TestCase):
    """Tests for SimpleDataset class"""

    def setUp(self):
        num_examples = 15
        # The matrix is an array of sequences of varying sizes. Each
        # sequence is an array of one element.
        self.train_instances = [
            numpy.array([[x, x+1] for x in range(sequence_length)])
            for sequence_length in random.sample(k=num_examples,
                                                 population=range(3, 20))]
        self.train_instances = numpy.array(self.train_instances)
        self.train_labels = (
            numpy.random.random((num_examples,)) * 10).astype(numpy.int16)

        self.test_instances = [
            numpy.array([[x*20, x*20+1] for x in range(sequence_length)])
            for sequence_length in random.sample(k=num_examples,
                                                 population=range(3, 20))]
        self.test_instances = numpy.array(
            self.test_instances + [numpy.array([[50, 60], [500, 600]])])
        self.test_labels = (
            numpy.random.random((num_examples + 1,)) * 10).astype(numpy.int16)

        # We ensure each label is at least three times
        self.partition_sizes = {
            'train': 0.5, 'test': 0.25, 'validation': 0.1
        }
        self.dataset = KDDCupDataset()
        self.dataset.create_fixed_samples(
            self.train_instances, self.train_labels,
            self.test_instances, self.test_labels,
            samples_num=1, partition_sizes=self.partition_sizes)
        self.dataset.set_current_sample(0)

    def test_test_split(self):
        """Test all the test instances come from the test dataset."""
        while self.dataset.has_next_batch(5, partition_name='test'):
            batch, labels, lengths = self.dataset.next_batch(
                batch_size=5, partition_name='train')
            for instance in batch:
                self.assertGreaterEqual(instance[instance.nonzero()].min(), 60)

        self.assertEqual(
            self.dataset.num_examples('test'),
            int(self.test_instances.shape[0] * self.partition_sizes['test']))

    def test_train_validation_split(self):
        """Test the train and validation instances come from the train dataset.
        """
        while self.dataset.has_next_batch(5, partition_name='train'):
            batch, labels, lengths = self.dataset.next_batch(
                batch_size=5, partition_name='train')
            for instance in batch:
                self.assertLessEqual(instance.max(), 22)

        while self.dataset.has_next_batch(5, partition_name='validation'):
            batch, labels, lengths = self.dataset.next_batch(
                batch_size=5, partition_name='train')
            for instance in batch:
                self.assertLessEqual(instance.max(), 22)

    def test_maximums(self):
        self.assertEqual(self.dataset.maximums.tolist(), [500, 600])
        self.assertEqual(self.dataset.maximums.shape[0],
                         self.dataset.feature_vector_size)


if __name__ == '__main__':
    unittest.main()