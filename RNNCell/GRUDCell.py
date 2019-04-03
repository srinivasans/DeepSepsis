"""
Author: Srinivasan Sivanandan
"""
import tensorflow as tf
from tensorflow.contrib.rnn import LayerRNNCell
from tensorflow.python.keras import activations
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import initializers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
import numpy as np

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "weights"

class GRUDCell(LayerRNNCell):
  """
  Implementation of GRUD cell as described in  (cf. https://www.nature.com/articles/s41598-018-24271-9).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().
  """

  def __init__(self,
               input_size,
               hidden_size,
               indicator_size,
               delta_size,
               output_size = 1,
               dropout_rate = 0.5,
               xMean = None,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               dtype=None,
               **kwargs):
    super(GRUDCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)

    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)

    self._num_units = input_size
    self._hidden_size = hidden_size
    self._indicator_size = indicator_size
    self._delta_size = delta_size
    self._output_size = output_size
    self._dropout_rate = dropout_rate

    self._xMean = xMean
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh
    self._kernel_initializer = initializers.get(kernel_initializer)
    self._bias_initializer = initializers.get(bias_initializer)

  @property
  def state_size(self):
    return self._hidden_size

  @property
  def output_size(self):
    return self._hidden_size

  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    #input_depth = inputs_shape[-1]
    self._decay_kernel = self.add_variable(
        "decay/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._delta_size, self._delta_size],
        initializer=self._kernel_initializer)
    self._decay_bias = self.add_variable(
        "decay/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._delta_size],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))

    self._decay_h_kernel = self.add_variable(
        "decay_h/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._delta_size, self._hidden_size],
        initializer=self._kernel_initializer)
    self._decay_h_bias = self.add_variable(
        "decay_h/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._hidden_size],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))

    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units + self._hidden_size + self._indicator_size, 2*self._hidden_size],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2*self._hidden_size],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        
    
    self._candidate_kernel = self.add_variable(
        "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units + self._hidden_size + self._indicator_size, self._hidden_size],
        initializer=self._kernel_initializer)
    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._hidden_size],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    X=inputs[:,0:self._num_units]
    m=inputs[:,self._num_units:self._num_units+self._indicator_size]
    delta=inputs[:,self._num_units+self._indicator_size:self._num_units+self._indicator_size+self._delta_size]

    # one more experiment TODO Use separate weights for x and h and train
    decay_kernel_diagonal = np.eye(self._delta_size)*self._decay_kernel
    decay_state_x = math_ops.matmul(delta, decay_kernel_diagonal)
    decay_state_x = nn_ops.bias_add(decay_state_x, self._decay_bias)

    decay_state_h = math_ops.matmul(delta, self._decay_h_kernel)
    decay_state_h = nn_ops.bias_add(decay_state_h, self._decay_h_bias)

    gamma_x = tf.math.exp(-tf.maximum(0.0,decay_state_x))
    gamma_h = tf.math.exp(-tf.maximum(0.0,decay_state_h))
    
    X_d = m*X + (1-m)*(gamma_x*X + (1-gamma_x)*self._xMean)
    state_d = gamma_h*state

    gate_inputs = math_ops.matmul(
        array_ops.concat([X_d, state_d, m], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state_d

    candidate = math_ops.matmul(
        array_ops.concat([X_d, r_state, m], 1), self._candidate_kernel)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = (1 - u) * state_d + u * c

    #new_h_drop = tf.nn.dropout(new_h,keep_prob=self._dropout_rate)
    #output = math_ops.matmul(new_h, self._output_kernel)
    #output = nn_ops.bias_add(output, self._output_bias)
    
    #output = math_ops.sigmoid(output)
    # Return logits to use tensorflow sigmoid_cross_entropy with logits function
    return new_h, new_h

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "hidden_size":self._hidden_size,
        "indicator_size":self._indicator_size,
        "delta_size":self._delta_size,
        "output_size": self._output_size,
        "kernel_initializer": initializers.serialize(self._kernel_initializer),
        "bias_initializer": initializers.serialize(self._bias_initializer),
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(GRUDCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
