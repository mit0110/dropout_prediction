import numpy
import random
import tensorflow as tf
import unittest

from kddcup_dataset import KDDCupDataset
from models.kdd_coembedded_lstm import KDDCupCoEmbeddedLSTMModel


class KDDCupCoEmbeddedLSTMModelTest(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        num_examples = 100
        # The matrix is an array of sequences of varying sizes. Each
        # sequence is an array of dimension 1.
        train_instances = numpy.array([numpy.arange(random.randint(3, 20))
                                       for _ in range(num_examples)])
        train_labels = numpy.round(numpy.random.random(
            (num_examples,))).astype(numpy.int32)

        test_instances = numpy.array([numpy.arange(random.randint(3, 20))
                                       for _ in range(num_examples)])
        test_labels = numpy.round(numpy.random.random(
            (num_examples,))).astype(numpy.int32)

        self.partition_sizes = {
            'train': 0.60, 'test': 1, 'validation': 0.15
        }
        self.dataset = KDDCupDataset()
        self.dataset.create_fixed_samples(
            train_instances, train_labels, test_instances, test_labels,
            samples_num=1, partition_sizes=self.partition_sizes)
        self.dataset.set_current_sample(0)
        self.model_arguments = {
            'hidden_layer_size': 50, 'batch_size': 20, 'logs_dirname': None,
            'log_values': 0, 'max_num_steps': 25}

    def test_build_network(self):
        """Test if the LSTMModel is correctly built."""
        # Check build does not raise errors
        model = KDDCupCoEmbeddedLSTMModel(self.dataset, **self.model_arguments)
        model.fit(close_session=True, training_epochs=50)

    def test_predict(self):
        """Test if the LSTMModel returns consistent predictions."""
        # Check build does not raise errors
        model = KDDCupCoEmbeddedLSTMModel(self.dataset, **self.model_arguments)
        model.fit(training_epochs=50)
        true, predictions = model.predict('test')
        expected_size = ((self.dataset.num_examples('test') //
                          model.batch_size) * model.batch_size)
        self.assertEqual(true.shape[0], expected_size)
        self.assertEqual(true.shape, predictions.shape)

    def test_evaluate(self):
        """Test if the LSTMModel returns a valid accuracy value."""
        # Check build does not raise errors
        model = KDDCupCoEmbeddedLSTMModel(self.dataset, **self.model_arguments)
        model.fit(training_epochs=50)
        metric = model.evaluate('test')
        self.assertLessEqual(0, metric)
        self.assertGreaterEqual(1, metric)

if __name__ == '__main__':
    unittest.main()