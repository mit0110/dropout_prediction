import numpy
import random
import tensorflow as tf
import unittest

from kddcup_dataset import KDDCupDataset
from models.kdd_lstm import (KDDCupLSTMModel, TruncKDDCupLSTMModel,
                             KDDCupBiLSTMModel)


class KDDCupLSTMModelTest(unittest.TestCase):

    MODEL_TO_TEST = KDDCupLSTMModel

    def setUp(self):
        tf.reset_default_graph()
        num_examples = 100
        # The matrix is an array of sequences of varying sizes. Each
        # sequence is an array of two elements.
        train_instances = numpy.array([
            numpy.array([numpy.array([x, x + 1])
                         for x in range(random.randint(3, 20))])
            for _ in range(num_examples)])
        train_labels = numpy.round(numpy.random.random(
            (num_examples,))).astype(numpy.int32)

        test_instances = numpy.array([
            numpy.array([numpy.array([x, x + 1])
                         for x in range(random.randint(3, 20))])
            for _ in range(num_examples)])
        test_labels = numpy.round(numpy.random.random(
            (num_examples,))).astype(numpy.int32)

        self.partition_sizes = {
            'train': 0.60, 'test': 0.25, 'validation': 0.15
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
        model = self.MODEL_TO_TEST(self.dataset, **self.model_arguments)
        model.fit(close_session=True)

    def test_predict(self):
        """Test if the LSTMModel returns consistent predictions."""
        # Check build does not raise errors
        model = self.MODEL_TO_TEST(self.dataset, **self.model_arguments)
        model.fit()
        true, predictions = model.predict('test')
        self.assertEqual(true.shape[0], self.dataset.num_examples('test'))
        self.assertEqual(true.shape, predictions.shape)

    def test_evaluate(self):
        """Test if the LSTMModel returns a valid rmse value."""
        # Check build does not raise errors
        model = self.MODEL_TO_TEST(self.dataset, **self.model_arguments)
        model.fit()
        metric = model.evaluate('test')
        self.assertLessEqual(0, metric)
        self.assertGreaterEqual(1, metric)


class TruncKDDCupLSTMModelTest(KDDCupLSTMModelTest):
    MODEL_TO_TEST = TruncKDDCupLSTMModel


class KDDCupBiLSTMModelTest(KDDCupLSTMModelTest):
    MODEL_TO_TEST = KDDCupBiLSTMModel


if __name__ == '__main__':
    unittest.main()