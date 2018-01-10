import tensorflow as tf

from models.kdd_embedded_lstm import KDDCupEmbeddedLSTMModel
from tensorflow.python.ops.math_ops import tanh


class EmbeddedBasicLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    """BasicLSTMCell to transform the input before running the cell."""

    def __init__(self, num_units, forget_bias=1.0,
                 state_is_tuple=True, activation=tanh, reuse=None,
                 modifier_function=None):
        super(EmbeddedBasicLSTMCell, self).__init__(
            num_units, forget_bias=forget_bias, state_is_tuple=state_is_tuple,
            activation=activation, reuse=reuse)
        self.modifier_function = modifier_function

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`.
            state: An `LSTMStateTuple` of state tensors, each shaped
                `[batch_size x self.state_size]`, if `state_is_tuple` has been
                set to `True`.  Otherwise, a `Tensor` shaped
                `[batch_size x 2 * self.state_size]`.
        Returns:
            A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        if self._state_is_tuple:
            c, h = state
        else:
            raise ValueError('EmbeddedBasicLSTMCell must use a state tuple')
        # TODO if self.modifier_function is not None:
        inputs = tf.abs(tf.subtract(inputs, h))
        return super(EmbeddedBasicLSTMCell, self).call(inputs, state)


class KDDCupCoEmbeddedLSTMModel(KDDCupEmbeddedLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer. The embedding_size will be the
    same as the hidden layer size.
    """
    def __init__(self, dataset, hidden_layer_size=0, **kwargs):
        super(KDDCupCoEmbeddedLSTMModel, self).__init__(
            dataset, hidden_layer_size=hidden_layer_size,
            embedding_size=hidden_layer_size, **kwargs)

    def _build_rnn_cell(self):
        return EmbeddedBasicLSTMCell(self.hidden_layer_size, forget_bias=1.0)


