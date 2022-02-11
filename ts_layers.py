# This code is to accompany the paper "Deep Learning in Target Space", M. Fairbank, S. Samothrakis, L. Citi, https://jmlr.org/papers/v23/20-040.html
# Please cite the above paper if this code or future variants of it are used in future academic work.
# Pull requests to improve or tidy this code up are welcome

from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.linalg import lstsq 

class TSLayer(Layer):
    def __init__(self,
                             realisation_batch_size, 
                             pseudoinverse_l2_regularisation,
                             activity_regularizer,
                             **kwargs):
        super(TSLayer, self).__init__(
                activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.realisation_batch_size=realisation_batch_size
        self.pseudoinverse_l2_regularisation=pseudoinverse_l2_regularisation

class TSDense(TSLayer):
    """A Target-space variety of the Keras Dense Layer.
    This code is to accompany paper https://jmlr.org/papers/v23/20-040.html

    Please cite the above paper if this code or future variants of it are used in future academic work.
    
    This code is (badly) modified from the Keras Dense implementation, and may 
    need further modifications to conform to keras standards.
    M. Fairbank 2020-06-03
    
    ```
    Arguments:
        units: Positive integer, dimensionality of the output space.
        target_input_matrix:  The activations corresponding to a fixed 
            representitive sample of the training input vectors.  
            See target-space paper section 3.1 for details.  If this is not the 
            first layer of the network, then it must be link to the previous 
            layer's calculate_target_output_matrix() function.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        pseudoinverse_l2_regularisation: The amount of L2 regularisation 
            to use in the least-squares solution.  See target-space paper 
            section 3.2. If in doubt, leave as default value 0.1
        shortcuts: If True, then appends inputs to outputs simulating shortcut
            weights from previous layer to next layer.
        use_scu_algorithm: Switches on preferred target-space algorithm, 
            sequential cascade correction (SCU).  Alternative is OCU algorithm.
            See target-space paper section 2.4 for details.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the targets matrix.  Defaults to stddev=1.
        initialise_with_projection:  Boolean (default True) to say if the 
            target matrices are further initialised by projecting them onto the 
            initial layer outputs. This is tentatively recommened behaviour.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                             units,
                             realisation_batch_size, 
                             pseudoinverse_l2_regularisation=0.1,
                             activation=None,
                             use_bias=True,
                             use_scu_algorithm=True, 
                             kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.),
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             kernel_constraint=None,
                             **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(TSDense, self).__init__(
            realisation_batch_size=realisation_batch_size, 
            pseudoinverse_l2_regularisation=pseudoinverse_l2_regularisation,
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.target_matrix = self.add_weight(
                'target_matrix',
                shape=(realisation_batch_size+[self.units]) if isinstance(realisation_batch_size, list) else [realisation_batch_size, self.units],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)    
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)
        self.use_scu_algorithm=use_scu_algorithm
                
    def calculate_internal_weight_matrix(self,target_input_matrix):
        # Convert the target matrix into an ordinary weight matrix, by solving 
        # the least squares problem of linearly-transforming
        # the target input-matrix into the target output matrix.
        b=self.target_matrix
        if isinstance(self.realisation_batch_size, list):
            target_input_matrix=tf.reshape(target_input_matrix,[-1,target_input_matrix.get_shape()[-1]])
            b=tf.reshape(b,[-1,b.get_shape()[-1]])
        if self.use_bias:
            # bias nodes need incorporating into the input matrix for the last-squares method to find the bias weights at the same time as the main weights.
            inputs_with_bias=tf.concat([tf.ones_like(target_input_matrix[:,0:1]),target_input_matrix],axis=1)
        else:
            inputs_with_bias=target_input_matrix  
        return lstsq(inputs_with_bias, b, l2_regularizer=self.pseudoinverse_l2_regularisation)

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.bias = None
        self.built = True
        
    def initialise_targets_with_projection(self, target_input_matrix):
        internal_weight_matrix=self.calculate_internal_weight_matrix(target_input_matrix)
        # This step during initialisation sometimes appears to make target 
        # layers train better.  It projects the targets onto the current 
        # actual layer sums. See section 3.4 of target space paper for details.
        output=self.propagate_layer(target_input_matrix, internal_weight_matrix, apply_activation=False)
        self.target_matrix.assign(output)
        # Now use these new targets to calculate the new weights to compute the actual new output
        internal_weight_matrix=self.calculate_internal_weight_matrix(target_input_matrix)
        output=self.propagate_layer(target_input_matrix, internal_weight_matrix, apply_activation=True)
        return output
    

    def call(self, target_input_matrix, inputs0):
        internal_weight_matrix=self.calculate_internal_weight_matrix(target_input_matrix)
        if self.use_scu_algorithm:
            # SCU (sequential cascade untangling) algorithm, defined in target-space paper Section 2.4, Algorithm 3.
            # This calculates exactly how well the targets have been met, so that any imperfections can be carried 
            # forwards to the next layer to be further corrected for.
            target_outputs=self.propagate_layer(target_input_matrix,internal_weight_matrix)
        else:
            # OCU algorithm defined in target space paper, optimistic cascade untangling; 
            # not usually as good, included merely to allow comparison.
            target_outputs=self.target_matrix
            if self.activation is not None:
                target_outputs=self.activation(target_outputs)
        main_output_matrix=self.propagate_layer(inputs0,internal_weight_matrix)
        return [target_outputs, main_output_matrix]
    
    def propagate_layer(self, inputs, internal_weight_matrix, apply_activation=True):
        if isinstance(self.realisation_batch_size, list):
            rank=len(self.realisation_batch_size)+1
        else:
            rank=2
        if self.use_bias:
            if rank==2:
                inputs_with_bias=tf.concat([tf.ones_like(inputs[:,0:1]),inputs],axis=1)
            elif rank==3:
                inputs_with_bias=tf.concat([tf.ones_like(inputs[:,:,0:1]),inputs],axis=2)
        else:
            inputs_with_bias=inputs
        if rank>2:
            internal_weight_matrix=tf.reshape(internal_weight_matrix,[1,internal_weight_matrix.get_shape()[0],internal_weight_matrix.get_shape()[1]])
        inputs_with_bias = tf.cast(inputs_with_bias, self._compute_dtype)
        if K.is_sparse(inputs):
            outputs = sparse_ops.sparse_tensor_dense_matmul(inputs_with_bias, internal_weight_matrix)
        else:
            outputs = tf.matmul(inputs_with_bias, internal_weight_matrix)
        if self.activation is not None and apply_activation:
            outputs=self.activation(outputs)
        return outputs    

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                    'The innermost dimension of input_shape must be defined, but saw: %s'
                    % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        # TODO this function is copied from keras Dense, and probably needs some further changes for TSDense...?
        config = {
                'units': self.units,
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                'activity_regularizer':
                        regularizers.serialize(self.activity_regularizer),
                'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(TSDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
class TSConv2D(TSLayer):
    # Copied and modifed from Keras.Conv2D layer.  Some code needs tidying up as a result
    """Abstract N-D convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Note: layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Arguments:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space (i.e. the number
            of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"`,    `"same"`, or `"causal"` (case-insensitive).
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch_size, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch_size, channels, ...)`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: An initializer for the convolution kernel.
        bias_initializer: An initializer for the bias vector. If None, the default
            initializer will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
                kernel after being updated by an `Optimizer` (e.g. used to implement
                norm constraints or value constraints for layer weights). The function
                must take as input the unprojected variable and must return the
                projected variable (which must have the same shape). Constraints are
                not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
                bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` the weights of this layer will be marked as
            trainable (and listed in `layer.trainable_weights`).
        name: A string, the name of the layer.
    """

    def __init__(self, 
                             realisation_batch_size,
                             filters,
                             kernel_size,
                             pseudoinverse_l2_regularisation=0.1,
                             regularisaton_relative_to_num_patches=False,
                             strides=1,
                             padding='valid',
                             data_format=None,
                             dilation_rate=1,
                             activation=None,
                             use_bias=True,
                             kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=.1),
                             bias_initializer='zeros',
                             kernel_regularizer=None,
                             bias_regularizer=None,
                             activity_regularizer=None,
                             kernel_constraint=None,
                             bias_constraint=None,
                             trainable=True,
                             name=None,
                             **kwargs):
        super(TSConv2D, self).__init__(
                realisation_batch_size=realisation_batch_size, 
                pseudoinverse_l2_regularisation=pseudoinverse_l2_regularisation,
                trainable=trainable,
                name=name,
                activity_regularizer=regularizers.get(activity_regularizer),
                **kwargs)
        rank=self.rank = 2
        if filters is not None and not isinstance(filters, int):
            filters = int(filters)
        self.filters = filters
        self.regularisaton_relative_to_num_patches=regularisaton_relative_to_num_patches
        self.kernel_size = conv_utils.normalize_tuple(
                kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if (self.padding == 'causal' and not isinstance(self,(Conv1D, SeparableConv1D))):
            raise ValueError('Causal padding is only supported for `Conv1D`'
                                             'and ``SeparableConv1D`.')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
                dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self._padding_op = self._get_padding_op()
        
    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel, self.filters)
        #print("kernel_shape",kernel_shape, kernel.shape,type(kernel_shape),type(kernel.shape))#(3,3,1,6)
        #sys.exit(0)
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(ndim=self.rank + 2,
                        axes={channel_axis: input_channel})
        self._build_conv_op_input_shape = input_shape
        self._build_input_channel = input_channel
        self._padding_op = self._get_padding_op()
        self._conv_op_data_format = conv_utils.convert_data_format(
                self.data_format, self.rank + 2)
        self._convolution_op = nn_ops.Convolution(
                input_shape,
                filter_shape=tensor_shape.TensorShape(kernel_shape),
                dilation_rate=self.dilation_rate,
                strides=self.strides,
                padding=self._padding_op,
                data_format=self._conv_op_data_format)
        self.image_height=input_shape[1] #.value
        self.image_width=input_shape[2] #.value
        self.input_channels=input_shape[3]
        num_patches=self.image_width*self.image_height
        self.num_patches=num_patches
        self.target_matrix = self.add_weight(
                'target_matrix',
                shape=[self.realisation_batch_size*num_patches, self.filters],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)    
        self.built = True

    def call(self, target_inputs, inputs, training=False):
        if self._recreate_conv_op(inputs):
            self._convolution_op = nn_ops.Convolution(
                    inputs.get_shape(),
                    filter_shape=self.kernel.shape,
                    dilation_rate=self.dilation_rate,
                    strides=self.strides,
                    padding=self._padding_op,
                    data_format=self._conv_op_data_format)

        # Apply causal padding to inputs for Conv1D.
        if self.padding == 'causal' and self.__class__.__name__ == 'Conv1D':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())

        b,W=self.calculate_internal_weight_matrix(target_inputs, training)
        outputs=self.propagate_layer(inputs, W,b)
        target_outputs=self.propagate_layer(target_inputs, W,b)
        return [target_outputs,outputs]
        
    def initialise_targets_with_projection(self, target_input_matrix):
        b,W=self.calculate_internal_weight_matrix(target_input_matrix,training=False)
        # This step during initialisation sometimes appears to make target 
        # layers train better.  It projects the targets onto the current 
        # actual layer sums. See section 3.4 of target space paper for details.
        output=self.propagate_layer(target_input_matrix, W, b, apply_activation=False)
        self.target_matrix.assign(tf.reshape(output,[-1,self.filters]))
        # Now use these new targets to calculate the new weights to compute the actual new output
        b,W=self.calculate_internal_weight_matrix(target_input_matrix,training=False)
        output=self.propagate_layer(target_input_matrix, W,b, apply_activation=True)
        return output
        
    def propagate_layer(self, input_image, kernel, bias, apply_activation=True):
        b,W=bias,kernel
        #print("Propagating layer. Image",input_image.shape, "W",W.shape)
        outputs = self._convolution_op(input_image, W)
        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(b, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, b, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, b, data_format='NHWC')
        if self.activation is not None and apply_activation:
            outputs=self.activation(outputs)
        return outputs

        
    def calculate_internal_weight_matrix(self, input_image, training):
        num_channels=input_image.get_shape()[3]#.value
        # Convert the target matrix into an ordinary weight matrix, by solving 
        # the least squares problem of linearly-transforming
        # the target input-matrix into the target output matrix.
        
        # Because this is a CNN layer, the rank-4 input tensor needs to have all 
        # its patches extracting and then flattening to form an input "matrix" 
        # which is suitable for the lstsq solution.
        input_patches=tf.image.extract_patches(images=input_image, sizes=[1,self.kernel_size[0],self.kernel_size[1],1], strides=[1,1,1,1], rates=[1,1,1,1], padding=self._padding_op)
        flattened_input=tf.reshape(input_patches,[-1,self.kernel_size[0]*self.kernel_size[1]*num_channels])
        if self.use_bias:
            # bias nodes need incorporating into the input matrix for the last-squares method to find the bias weights at the same time as the main weights.
            flattened_input=tf.concat([tf.ones_like(flattened_input[:,0:1]),flattened_input],axis=1)
        flattened_kernel= lstsq(flattened_input, self.target_matrix, self.pseudoinverse_l2_regularisation)

        if self.use_bias:
            b,W=flattened_kernel[0,:],flattened_kernel[1:,:]
        else:
            b,W=None,flattened_kernel
        W=tf.reshape(W,[self.kernel_size[0],self.kernel_size[1], num_channels, self.filters])
        return [b,W]

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding=self.padding,
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                                                            [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding=self.padding,
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                                                            new_space)

    def get_config(self):
        config = {
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'data_format': self.data_format,
                'dilation_rate': self.dilation_rate,
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'activity_regularizer':
                        regularizers.serialize(self.activity_regularizer),
                'kernel_constraint': constraints.serialize(self.kernel_constraint),
                'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return 1
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding

    def _recreate_conv_op(self, inputs):
        """Recreate conv_op if necessary.

        Check if the input_shape in call() is different from that in build().
        For the values that are not None, if they are different, recreate
        the _convolution_op to avoid the stateful behavior.

        Args:
            inputs: The input data to call() method.

        Returns:
            `True` or `False` to indicate whether to recreate the conv_op.
        """
        call_input_shape = inputs.get_shape()
        for axis in range(1, len(call_input_shape)):
            if (call_input_shape[axis] is not None
                    and self._build_conv_op_input_shape[axis] is not None
                    and call_input_shape[axis] != self._build_conv_op_input_shape[axis]):
                return True
        return False


class TSRNNDense(TSLayer):
    """A Target-space variety of the Keras Dense Layer.
    This code is to accompany "Deep Learning in Target Space", M. Fairbank + S. Samothrakis, arXiv:2006.01578
    Please cite the above paper if this code or future variants of it are used in future academic work.
    
    This code is (badly) modified from the Keras Dense implementation, and may 
    need further modifications to conform to keras standards.
    M. Fairbank 2020-06-03
    
    ```
    Arguments:
        units: Positive integer, dimensionality of the output space.
        target_input_matrix:  The activations corresponding to a fixed 
            representitive sample of the training input vectors.  
            See target-space paper section 3.1 for details.  If this is not the 
            first layer of the network, then it must be link to the previous 
            layer's calculate_target_output_matrix() function.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        pseudoinverse_l2_regularisation: The amount of L2 regularisation 
            to use in the least-squares solution.  See target-space paper 
            section 3.2. If in doubt, leave as default value 0.1
        shortcuts: If True, then appends inputs to outputs simulating shortcut
            weights from previous layer to next layer.
        use_scu_algorithm: Switches on preferred target-space algorithm, 
            sequential cascade correction (SCU).  Alternative is OCU algorithm.
            See target-space paper section 2.4 for details.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the targets matrix.  Defaults to stddev=1.
        initialise_with_projection:  Boolean (default True) to say if the 
            target matrices are further initialised by projecting them onto the 
            initial layer outputs. This is tentatively recommened behaviour.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                             units,
                             seq_length,
                             realisation_batch_size, 
                             realisation_seq_length,
                             pseudoinverse_l2_regularisation=0.1,
                             return_sequences=False,
                             activation=None,
                             use_bias=True,
                             kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.),
                             kernel_regularizer=None,
                             activity_regularizer=None,
                             kernel_constraint=None,
                             **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TSRNNDense, self).__init__(
            realisation_batch_size=realisation_batch_size, 
            pseudoinverse_l2_regularisation=pseudoinverse_l2_regularisation,
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.realisation_seq_length=realisation_seq_length 
        self.seq_length=seq_length 
        self.return_sequences=return_sequences
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.target_matrix = self.add_weight(
                'target_matrix',
                shape=[realisation_batch_size, realisation_seq_length, self.units],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)    
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)
                
    def calculate_internal_weight_matrix(self,target_input_matrix):
        # Convert the target matrix into an ordinary weight matrix, by solving 
        # the least squares problem of linearly-transforming
        # the target input-matrix into the target output matrix.

        estimated_output_matrix=self.target_matrix
        if self.activation is not None:
            estimated_output_matrix = self.activation(estimated_output_matrix)
        zeros_input_matrix=tf.zeros_like(estimated_output_matrix[:,0:1,:])
        
        estimated_recurrent_input_matrix=tf.concat([zeros_input_matrix,estimated_output_matrix[:,:-1,:]],axis=1)
        estimated_full_input_matrix=tf.concat([target_input_matrix,estimated_recurrent_input_matrix],axis=2)
        if self.use_bias:
            # bias nodes need incorporating into the input matrix for the last-squares method to find the bias weights at the same time as the main weights.
            estimated_full_input_matrix=tf.concat([tf.ones_like(estimated_full_input_matrix[:,:,0:1]),estimated_full_input_matrix],axis=2)
        A=tf.reshape(estimated_full_input_matrix,[-1,estimated_full_input_matrix.get_shape()[2]])
        b=tf.reshape(self.target_matrix,[-1,self.target_matrix.get_shape()[2]])
        full_kernel=lstsq(A, b, l2_regularizer=self.pseudoinverse_l2_regularisation)
        return full_kernel

    def propagate_layer(self, input_matrix, full_kernel_matrix, seq_length, return_sequences, apply_activation=True):
        if self.use_bias:
            input_matrix=tf.concat([tf.ones_like(input_matrix[:,:,0:1]),input_matrix],axis=2)
        x=tf.fill([tf.shape(input_matrix)[0], self.units], 0.0)
        result_list=[] 
        for t in range(seq_length):
            x=tf.concat([input_matrix[:,t,:],x],axis=1)
            x=tf.matmul(x,full_kernel_matrix)
            x_act = self.activation(x) if self.activation is not None else x
            result_list.append(x_act if apply_activation else x)
            x=x_act
        if return_sequences:
            return tf.stack(result_list,axis=1)
        else:
            return result_list[-1]

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.bias = None
        self.built = True
        
    def initialise_targets_with_projection(self, target_input_matrix):
        internal_weight_matrix=self.calculate_internal_weight_matrix(target_input_matrix)
        # This step during initialisation sometimes appears to make target 
        # layers train better.  It projects the targets onto the current 
        # actual layer sums. See section 3.4 of target space paper for details.
        output=self.propagate_layer(target_input_matrix, internal_weight_matrix, self.realisation_seq_length, return_sequences=True,apply_activation=False)
        self.target_matrix.assign(output)
        # Now use these new targets to calculate the new weights to compute the actual new output
        internal_weight_matrix=self.calculate_internal_weight_matrix(target_input_matrix)
        output=self.propagate_layer(target_input_matrix, internal_weight_matrix, self.realisation_seq_length, return_sequences=True,apply_activation=True)
        if not self.return_sequences:
            output=output[:,-1,:]
        return output
    

    def call(self, target_input_matrix,inputs):
        full_kernel=self.calculate_internal_weight_matrix(target_input_matrix)
        output_matrix=self.propagate_layer(inputs, full_kernel, self.seq_length, self.return_sequences)
        target_output_matrix=self.propagate_layer(target_input_matrix, full_kernel, self.realisation_seq_length, self.return_sequences)
        return target_output_matrix,output_matrix

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                    'The innermost dimension of input_shape must be defined, but saw: %s'
                    % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        # TODO this function is copied from keras Dense, and probably needs some further changes for TSDense...?
        config = {
                'units': self.units,
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                'activity_regularizer':
                        regularizers.serialize(self.activity_regularizer),
                'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(TSDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
   
