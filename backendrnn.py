import tensorflow as tf


from keras.utils import control_flow_util

def reverse(x, axes):
  """Reverse a tensor along the specified axes.

  Args:
      x: Tensor to reverse.
      axes: Integer or iterable of integers.
          Axes to reverse.

  Returns:
      A tensor.
  """
  if isinstance(axes, int):
    axes = [axes]
  return tf.reverse(x, axes)

def expand_dims(x, axis=-1):
  """Adds a 1-sized dimension at index "axis".

  Args:
      x: A tensor or variable.
      axis: Position where to add a new axis.

  Returns:
      A tensor with expanded dimensions.
  """
  return tf.expand_dims(x, axis)

def zeros_like(x, dtype=None, name=None):
  """Instantiates an all-zeros variable of the same shape as another tensor.

  Args:
      x: Keras variable or Keras tensor.
      dtype: dtype of returned Keras variable.
             `None` uses the dtype of `x`.
      name: name for the variable to create.

  Returns:
      A Keras variable with the shape of `x` filled with zeros.

  Example:

  ```python
  from tensorflow.keras import backend as K
  kvar = K.variable(np.random.random((2,3)))
  kvar_zeros = K.zeros_like(kvar)
  K.eval(kvar_zeros)
  # array([[ 0.,  0.,  0.], [ 0.,  0.,  0.]], dtype=float32)
  ```
  """
  return tf.zeros_like(x, dtype=dtype, name=name)

def rnn(step_function,
        inputs,
        initial_states,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False,
        input_length=None,
        time_major=False,
        zero_output_for_mask=False):
  """Iterates over the time dimension of a tensor.

  """

  def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return tf.compat.v1.transpose(input_t, axes)

  if not time_major:
    inputs = tf.nest.map_structure(swap_batch_timestep, inputs)

  flatted_inputs = tf.nest.flatten(inputs)
  time_steps = flatted_inputs[0].shape[0]
  batch = flatted_inputs[0].shape[1]
  time_steps_t = tf.shape(flatted_inputs[0])[0]

  for input_ in flatted_inputs:
    input_.shape.with_rank_at_least(3)

  if mask is not None:
    if mask.dtype != tf.bool:
      mask = tf.cast(mask, tf.bool)
    if len(mask.shape) == 2:
      mask = expand_dims(mask)
    if not time_major:
      mask = swap_batch_timestep(mask)

  if constants is None:
    constants = []

  # tf.where needs its condition tensor to be the same shape as its two
  # result tensors, but in our case the condition (mask) tensor is
  # (nsamples, 1), and inputs are (nsamples, ndimensions) or even more.
  # So we need to broadcast the mask to match the shape of inputs.
  # That's what the tile call does, it just repeats the mask along its
  # second dimension n times.
  def _expand_mask(mask_t, input_t, fixed_dim=1):
    if tf.nest.is_nested(mask_t):
      raise ValueError('mask_t is expected to be tensor, but got %s' % mask_t)
    if tf.nest.is_nested(input_t):
      raise ValueError('input_t is expected to be tensor, but got %s' % input_t)
    rank_diff = len(input_t.shape) - len(mask_t.shape)
    for _ in range(rank_diff):
      mask_t = tf.expand_dims(mask_t, -1)
    multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]
    return tf.tile(mask_t, multiples)

  if unroll:
    if not time_steps:
      raise ValueError('Unrolling requires a fixed number of timesteps.')
    states = tuple(initial_states)
    successive_states = []
    successive_outputs = []

    # Process the input tensors. The input tensor need to be split on the
    # time_step dim, and reverse if go_backwards is True. In the case of nested
    # input, the input is flattened and then transformed individually.
    # The result of this will be a tuple of lists, each of the item in tuple is
    # list of the tensor with shape (batch, feature)
    def _process_single_input_t(input_t):
      input_t = tf.unstack(input_t)  # unstack for time_step dim
      if go_backwards:
        input_t.reverse()
      return input_t

    if tf.nest.is_nested(inputs):
      processed_input = tf.nest.map_structure(_process_single_input_t, inputs)
    else:
      processed_input = (_process_single_input_t(inputs),)

    def _get_input_tensor(time):
      inp = [t_[time] for t_ in processed_input]
      return tf.nest.pack_sequence_as(inputs, inp)

    if mask is not None:
      mask_list = tf.unstack(mask)
      if go_backwards:
        mask_list.reverse()

      for i in range(time_steps):
        inp = _get_input_tensor(i)
        mask_t = mask_list[i]
        output, new_states = step_function(inp,
                                           tuple(states) + tuple(constants), i)
        tiled_mask_t = _expand_mask(mask_t, output)

        if not successive_outputs:
          prev_output = zeros_like(output)
        else:
          prev_output = successive_outputs[-1]

        output = tf.where(tiled_mask_t, output, prev_output)

        flat_states = tf.nest.flatten(states)
        flat_new_states = tf.nest.flatten(new_states)
        tiled_mask_t = tuple(_expand_mask(mask_t, s) for s in flat_states)
        flat_final_states = tuple(
            tf.where(m, s, ps)
            for m, s, ps in zip(tiled_mask_t, flat_new_states, flat_states))
        states = tf.nest.pack_sequence_as(states, flat_final_states)

        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = tf.stack(successive_outputs)

      if zero_output_for_mask:
        last_output = tf.where(
            _expand_mask(mask_list[-1], last_output), last_output,
            zeros_like(last_output))
        outputs = tf.where(
            _expand_mask(mask, outputs, fixed_dim=2), outputs,
            zeros_like(outputs))

    else:  # mask is None
      for i in range(time_steps):
        inp = _get_input_tensor(i)
        output, states = step_function(inp, tuple(states) + tuple(constants), i)
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = tf.stack(successive_outputs)

  else:  # Unroll == False
      raise ValueError('must be unroll')


  # static shape inference
  def set_shape(output_):
    if isinstance(output_, tf.Tensor):
      shape = output_.shape.as_list()
      shape[0] = time_steps
      shape[1] = batch
      output_.set_shape(shape)
    return output_

  outputs = tf.nest.map_structure(set_shape, outputs)

  if not time_major:
    outputs = tf.nest.map_structure(swap_batch_timestep, outputs)

  return last_output, outputs, new_states