import tensorflow as tf

from models.kdd_lstm import KDDCupLSTMModel


class KDDCupEmbeddedLSTMModel(KDDCupLSTMModel):
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
        dropout_ratio (float): the probability of dropout. It must range
            between 0 and 1.
        embedding_size (int): the number of units in the embedding layer.
        embedding_model (gensim.models.Word2Vec): if not None, initialize the
            embedding variable with the weights of the model.
    """

    def __init__(self, dataset, name=None, hidden_layer_size=0, batch_size=None,
                 logs_dirname='.', log_values=100, max_num_steps=100,
                 dropout_ratio=0.3, embedding_size=100, embedding_model=None):
        super(KDDCupEmbeddedLSTMModel, self).__init__(
            dataset, batch_size=batch_size, logs_dirname=logs_dirname,
            name=name, log_values=log_values, dropout_ratio=dropout_ratio,
            hidden_layer_size=hidden_layer_size, max_num_steps=max_num_steps)
        self.embedding_size = embedding_size
        self.embedding_var = None
        self.embedding_model = embedding_model

    def _build_inputs(self):
        """Generate placeholder variables to represent the input tensors."""
        # Placeholder for the inputs in a given iteration.
        self.instances_placeholder = tf.placeholder(
            tf.int32, (self.batch_size, self.max_num_steps),
            name='sequences_placeholder')

        self.lengths_placeholder = tf.placeholder(
            tf.int32, (self.batch_size, ), name='lengths_placeholder')

        self.labels_placeholder = tf.placeholder(
            self.dataset.labels_type, (self.batch_size, ),
            name='labels_placeholder')

        self.dropout_placeholder = tf.placeholder_with_default(
            0.0, shape=(), name='dropout_placeholder')

    def _build_input_layers(self):
        """Converts the instances_placeholder to an embedding.

        Returns:
            A tensor with shape [batch_size, max_num_steps, self.embedding_size]
        """
        # The sequences must be padded with a negative value, so the one
        # hot encoder generates a zero vector.
        if self.embedding_model is None:
            self.embedding_var = tf.Variable(
                tf.random_uniform([self.dataset.maximums + 1,
                                   self.embedding_size], 0, 1.0),
                trainable=True, name='input_embedding_var')
        else:
            embedding_matrix = self.embedding_model.wv.syn0
            # https://github.com/dennybritz/cnn-text-classification-tf/issues/17
            self.embedding_placeholder = tf.placeholder_with_default(
                embedding_matrix, shape=embedding_matrix.shape,
                name='embedding_placeholder')
            embedding_var = tf.Variable(tf.random_uniform(
                embedding_matrix.shape, -1.0, 1.0),
                name='input_embedding_var')
            self.embedding_init = embedding_var.assign(
                self.embedding_placeholder)
            # We add the embedding for the zero element, which SHOULD be the
            # padding element, and the embedding for the OOV element.
            self.embedding_var = tf.concat([
                tf.zeros([1, self.embedding_size]),
                embedding_var,
                tf.random_uniform([1, self.embedding_size], -1.0, 1.0)
            ], 0)
        return tf.nn.embedding_lookup(
            self.embedding_var, self.instances_placeholder,
            name='embedded_element_op')

    def build_all(self):
        super(KDDCupEmbeddedLSTMModel, self).build_all()
        if self.embedding_model is not None:
            with self.graph.as_default():
                self.sess.run([self.embedding_init])



