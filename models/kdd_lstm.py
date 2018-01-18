import tensorflow as tf

from quick_experiment.models.lstm import LSTMModel
from quick_experiment.models.bi_lstm import BiLSTMModel
from quick_experiment.models.lstm_tbptt import TruncLSTMModel


class KDDCupLSTMModel(LSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts a single output for a sequence.

    The inputs are expected to be of type int. Each column will be converted
    to a one hot encoding.

    Args:
        dataset (:obj: SequenceDataset): An instance of KDDCupDataset (or
            subclass). The dataset MUST have partition called validation.
        hidden_layer_size (int): The size of the hidden layer of the network.
        batch_size (int): The maximum size of elements to input into the model.
            It will also be used to generate batches from the dataset.
        logs_dirname (string): Name of directory to save internal information
            for tensorboard visualization. If None, no records will be saved
        log_values (int): Number of steps to wait before logging the progress
            of the training in console. If 0, no logs will be generated.
        max_num_steps (int): the maximum number of steps to use during the
            Back Propagation Through Time optimization. The gradients are
            going to be clipped at max_num_steps.
    """
    def _build_inputs(self):
        """Generate placeholder variables to represent the input tensors."""
        # Placeholder for the inputs in a given iteration.
        self.instances_placeholder = tf.placeholder(
            tf.int32, (None, self.max_num_steps,
                       self.dataset.feature_vector_size),
            name='sequences_placeholder')

        self.lengths_placeholder = tf.placeholder(
            tf.int32, (None, ), name='lengths_placeholder')

        self.labels_placeholder = tf.placeholder(
            self.dataset.labels_type, (None, ),
            name='labels_placeholder')

    def _build_input_layers(self):
        """Converts each column of instances_placeholder to a one hot encoding.
        """
        # The sequences must be padded with a negative value, so the one
        # hot encoder generates a zero vector.
        self.dropout_placeholder = tf.placeholder_with_default(
            0.0, shape=(), name='dropout_placeholder')

        one_hot_columns = []
        input_tensor = self._pad_batch(self.instances_placeholder)
        for column_position, column_max in enumerate(self.dataset.maximums):
            column = tf.slice(input_tensor,
                              begin=[0, 0, column_position],
                              size=[-1, -1, 1])
            one_hot_columns.append(tf.one_hot(column, depth=column_max,
                                              on_value=1, off_value=0))
        return tf.squeeze(tf.concat(one_hot_columns, axis=-1))

    def _build_predictions(self, logits):
        """Return a tensor with the predicted dropout.

        The prediction is a float in [0,1) with the probability of dropout.

        Args:
            logits: Tensor with unscaled logits, float - [batch_size, 2].

        Returns:
            A float64 tensor with the predictions, with shape [batch_size,].
        """
        predictions = tf.nn.softmax(logits)
        return tf.squeeze(predictions[:,1], name='predictions')

    def _build_evaluation(self, predictions):
        """Evaluate the quality of the logits at predicting the label.

        Args:
            predictions: Predictions tensor, int - [current_batch_size,
                max_num_steps].
        Returns:
            A scalar int32 tensor with the number of examples (out of
            batch_size) that were predicted correctly.
        """
        # predictions has shape [batch_size, ]
        with tf.name_scope('evaluation_performance'):
            mse, mse_update = tf.contrib.metrics.streaming_mean_squared_error(
                predictions,
                tf.cast(self.labels_placeholder, predictions.dtype))

        if self.logs_dirname:
            tf.summary.scalar('eval_mse', mse)
            tf.summary.scalar('eval_up_mse', mse_update)

        return mse, mse_update


class TruncKDDCupLSTMModel(KDDCupLSTMModel, TruncLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts a single output for a sequence.

    The inputs are expected to be of type int. Each column will be converted
    to a one hot encoding.

    Args:
        dataset (:obj: SequenceDataset): An instance of KDDCupDataset (or
            subclass). The dataset MUST have partition called validation.
        hidden_layer_size (int): The size of the hidden layer of the network.
        batch_size (int): The maximum size of elements to input into the model.
            It will also be used to generate batches from the dataset.
        logs_dirname (string): Name of directory to save internal information
            for tensorboard visualization. If None, no records will be saved
        log_values (int): Number of steps to wait before logging the progress
            of the training in console. If 0, no logs will be generated.
        max_num_steps (int): the maximum number of steps to use during the
            Back Propagation Through Time optimization. The gradients are
            going to be clipped at max_num_steps.
    """

    def evaluate(self, partition='validation'):
        old_start = self.dataset.reset_batch(partition)
        with self.graph.as_default():
            # Reset the metric variables
            stream_vars = [i for i in tf.local_variables()
                           if i.name.split('/')[0] == 'evaluation_performance']
            mse_op, mse_update_op = self.evaluation_op
            self.sess.run([tf.variables_initializer(stream_vars)])
            while self.dataset.has_next_batch(self.batch_size, partition):
                for feed_dict in self._fill_feed_dict(partition,
                                                      reshuffle=False):
                    feed_dict[self.dropout_placeholder] = 0
                    self.sess.run([mse_update_op], feed_dict=feed_dict)
            mse_value = self.sess.run([mse_op])[0]
        self.dataset.reset_batch(partition, old_start)
        return mse_value


class KDDCupBiLSTMModel(KDDCupLSTMModel, BiLSTMModel):
    pass