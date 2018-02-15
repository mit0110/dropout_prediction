import tensorflow as tf

from quick_experiment.models.bi_lstm import BiLSTMModel
from models.kdd_lstm import KDDCupLSTMModel

from tensorflow.contrib.tensorboard.plugins import projector


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

    def __init__(self, dataset, embedding_size=100, embedding_model=None,
                 finetune_embeddings=True, **kwargs):
        super(KDDCupEmbeddedLSTMModel, self).__init__(
            dataset, **kwargs)
        self.embedding_size = embedding_size
        self.embedding_model = embedding_model
        self.finetune_embeddings = finetune_embeddings
        self.embedding_var = None

    def _build_inputs(self):
        """Generate placeholder variables to represent the input tensors."""
        # Placeholder for the inputs in a given iteration.
        self.instances_placeholder = tf.placeholder(
            tf.int32, (None, self.max_num_steps),
            name='sequences_placeholder')

        self.lengths_placeholder = tf.placeholder(
            tf.int32, (None, ), name='lengths_placeholder')

        self.labels_placeholder = tf.placeholder(
            self.dataset.labels_type, (None, ),
            name='labels_placeholder')

        self.dropout_placeholder = tf.placeholder_with_default(
            0.0, shape=(), name='dropout_placeholder')

    def _pad_batch(self, input_tensor):
        self.current_batch_size = tf.shape(input_tensor)[0]
        new_instances = tf.subtract(self.batch_size, tf.shape(input_tensor)[0])
        # Pad lenghts
        self.batch_lengths = tf.pad(self.lengths_placeholder,
                                    paddings=[[tf.constant(0), new_instances]],
                                    mode='CONSTANT')
        # Pad instances
        paddings = [[tf.constant(0), new_instances], tf.constant([0, 0])]
        input_tensor = tf.pad(input_tensor, paddings=paddings, mode='CONSTANT')
        # Ensure the correct shape. This is only to avoid an error with the
        # dynamic_rnn, which needs to know the size of the batch.
        return tf.reshape(
            input_tensor, shape=(self.batch_size, self.max_num_steps))

    def _build_input_layers(self):
        """Converts the instances_placeholder to an embedding.

        Returns:
            A tensor with shape [batch_size, max_num_steps, self.embedding_size]
        """
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
                name='input_embedding_var', trainable=self.finetune_embeddings)
            self.embedding_init = embedding_var.assign(
                self.embedding_placeholder)
            # We add the embedding for the zero element, which SHOULD be the
            # padding element, and the embedding for the OOV element.
            self.embedding_var = tf.concat([
                tf.zeros([1, self.embedding_size]),
                embedding_var,
                tf.random_uniform([1, self.embedding_size], -1.0, 1.0)
            ], 0)
        input_tensor = self._pad_batch(self.instances_placeholder)
        return tf.nn.embedding_lookup(self.embedding_var, input_tensor,
                                      name='embedded_element_op')

    def build_all(self):
        super(KDDCupEmbeddedLSTMModel, self).build_all()
        if self.embedding_model is not None:
            with self.graph.as_default():
                self.sess.run([self.embedding_init])

    def write_embeddings(self, metadata_path):
        with self.graph.as_default():
            config = projector.ProjectorConfig()

            # Add positive embedding
            embedding = config.embeddings.add()
            embedding.tensor_name = self.embedding_var.name
            # Link this tensor to the same metadata file
            embedding.metadata_path = metadata_path

            # Saves a configuration file that TensorBoard will read
            # during startup.
            projector.visualize_embeddings(self.summary_writer, config)


class KDDCupEmbedBiLSTMModel(KDDCupEmbeddedLSTMModel, BiLSTMModel):
    pass
