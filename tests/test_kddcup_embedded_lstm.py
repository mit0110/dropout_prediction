import numpy
import random
import tensorflow as tf
import unittest

from gensim.models import Word2Vec
from kddcup_dataset import KDDCupDataset
from models.kdd_embedded_lstm import (KDDCupEmbeddedLSTMModel,
                                      KDDCupEmbedBiLSTMModel)


class KDDCupEmbeddedLSTMModelTest(unittest.TestCase):

    MODEL = KDDCupEmbeddedLSTMModel

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
        self.model_arguments = {
            'hidden_layer_size': 50, 'batch_size': 20, 'logs_dirname': None,
            'log_values': 0, 'max_num_steps': 25, 'embedding_size': 15}
        self.data = train_instances, train_labels, test_instances, test_labels

    def test_build_network(self):
        """Test if the LSTMModel is correctly built."""
        # Check build does not raise errors
        dataset = KDDCupDataset()
        dataset.create_fixed_samples(
            *self.data, samples_num=1, partition_sizes=self.partition_sizes)
        dataset.set_current_sample(0)
        model = self.MODEL(dataset, **self.model_arguments)
        model.fit(close_session=True, training_epochs=50)

    def test_build_network_no_finetuning(self):
        """Test if the LSTMModel is correctly built."""
        # Check build does not raise errors
        sentences = [[str(x) for x in numpy.arange(random.randint(3, 20))]
                     for _ in range(25)]
        embedding_model = Word2Vec(
            sentences=sentences, size=self.model_arguments['embedding_size'],
            iter=5)
        dataset = KDDCupDataset(embedding_model=embedding_model)
        dataset.create_fixed_samples(
            *self.data, samples_num=1, partition_sizes=self.partition_sizes)
        dataset.set_current_sample(0)
        # Check build does not raise errors

        model = self.MODEL(
            dataset, finetune_embeddings=False, embedding_model=embedding_model,
            **self.model_arguments)
        model.build_all()
        resulting_embeddings = model.sess.run(model.embedding_var)
        numpy.testing.assert_array_equal(resulting_embeddings[1:-1],
                                         embedding_model.wv.syn0)
        model.fit(training_epochs=50)
        resulting_embeddings = model.sess.run(model.embedding_var)
        numpy.testing.assert_array_equal(resulting_embeddings[1:-1],
                                         embedding_model.wv.syn0)

    def test_predict(self):
        """Test if the LSTMModel returns consistent predictions."""
        # Check build does not raise errors
        dataset = KDDCupDataset()
        dataset.create_fixed_samples(
            *self.data, samples_num=1, partition_sizes=self.partition_sizes)
        dataset.set_current_sample(0)
        model = self.MODEL(dataset, **self.model_arguments)
        model.fit(training_epochs=50)
        true, predictions = model.predict('test')
        expected_size = ((dataset.num_examples('test') //
                          model.batch_size) * model.batch_size)
        self.assertEqual(true.shape[0], expected_size)
        self.assertEqual(true.shape, predictions.shape)

    def test_evaluate(self):
        """Test if the LSTMModel returns a valid accuracy value."""
        dataset = KDDCupDataset()
        dataset.create_fixed_samples(
            *self.data, samples_num=1, partition_sizes=self.partition_sizes)
        dataset.set_current_sample(0)
        # Check build does not raise errors
        model = self.MODEL(dataset, **self.model_arguments)
        model.fit(training_epochs=50)
        metric = model.evaluate('test')
        self.assertLessEqual(0, metric)
        self.assertGreaterEqual(1, metric)

    def test_build_with_embeddings(self):
        """Test if the LSTMModel is correctly built."""
        # Train a very small model
        sentences = [[str(x) for x in numpy.arange(random.randint(3, 20))]
                     for _ in range(25)]
        embedding_model = Word2Vec(
            sentences=sentences, size=self.model_arguments['embedding_size'],
            iter=5)
        dataset = KDDCupDataset(embedding_model=embedding_model)
        dataset.create_fixed_samples(
            *self.data, samples_num=1, partition_sizes=self.partition_sizes)
        dataset.set_current_sample(0)
        # Check build does not raise errors
        model = self.MODEL(
            dataset, embedding_model=embedding_model,
            **self.model_arguments)
        model.build_all()
        resulting_embeddings = model.sess.run(model.embedding_var)
        numpy.testing.assert_array_equal(resulting_embeddings[1:-1],
                                         embedding_model.wv.syn0)
        model.fit(training_epochs=50)
        resulting_embeddings = model.sess.run(model.embedding_var)
        # No fine tuning, so it should change the embedding var,
        self.assertFalse(numpy.array_equal(resulting_embeddings[1:-1],
                                           embedding_model.wv.syn0))


class KDDCupBiEmbedLSTMModelTest(KDDCupEmbeddedLSTMModelTest):
    MODEL = KDDCupEmbedBiLSTMModel


if __name__ == '__main__':
    unittest.main()
