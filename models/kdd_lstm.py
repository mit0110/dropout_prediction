import tensorflow as tf

from quick_experiment.models.lstm import LSTMModel


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

    def __init__(self, dataset, name=None, hidden_layer_size=0, batch_size=None,
                 logs_dirname='.', log_values=100, max_num_steps=100,
                 dropout_ratio=0.3):
        super(KDDCupLSTMModel, self).__init__(
            dataset, batch_size=batch_size, logs_dirname=logs_dirname,
            name=name, log_values=log_values, dropout_ratio=dropout_ratio,
            hidden_layer_size=hidden_layer_size, max_num_steps=max_num_steps)

    def _build_inputs(self):
        """Generate placeholder variables to represent the input tensors."""
        # Placeholder for the inputs in a given iteration.
        self.instances_placeholder = tf.placeholder(
            tf.int32, (self.batch_size, self.max_num_steps,
                       self.dataset.feature_vector_size),
            name='sequences_placeholder')

        self.lengths_placeholder = tf.placeholder(
            tf.int32, (self.batch_size, ), name='lengths_placeholder')

        self.labels_placeholder = tf.placeholder(
            self.dataset.labels_type, (self.batch_size, ),
            name='labels_placeholder')

        self.dropout_placeholder = tf.placeholder_with_default(
            0.0, shape=(), name='dropout_placeholder')

    def _build_input_layers(self):
        """Converts each column of instances_placeholder to a one hot encoding.
        """
        # The sequences must be padded with a negative value, so the one
        # hot encoder generates a zero vector.
        one_hot_columns = []
        for column_position, column_max in enumerate(self.dataset.maximums):
            column = tf.slice(self.instances_placeholder,
                              begin=[0, 0, column_position],
                              size=[-1, -1, 1])
            #column = tf.Print(column, [tf.shape(column)])
            one_hot_columns.append(tf.one_hot(column, depth=column_max,
                                              on_value=1, off_value=0))
        return tf.squeeze(tf.concat(one_hot_columns, axis=-1))
