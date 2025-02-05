import keras
from skyrmion_dataset import SKYRMION
from functools import partial
import argparse
     
@keras.saving.register_keras_serializable()
class Model5(keras.Model):
    def __init__(self, args, **kwargs):

        self.args = args

        # Dense layer
        self.dense = partial(
            keras.layers.Dense,
            activation=keras.layers.Activation(self.args.activation),
            use_bias=True, 
            kernel_initializer=keras.initializers.glorot_normal(), 
            bias_initializer=keras.initializers.glorot_uniform(), 
            kernel_regularizer=keras.regularizers.l2(self.args.kernel_regularizer), 
            bias_regularizer=keras.regularizers.l2(self.args.bias_regularizer)
        )

        # Convolutional layer
        if self.args.conv_type == "standard":
            self.conv = partial(
                keras.layers.Conv2D,
                kernel_size=args.kernel_size,
                activation=keras.layers.Activation(self.args.activation),
                strides=self.args.stride,
                padding=self.args.padding,
                kernel_initializer=keras.initializers.he_normal(), 
                bias_initializer=keras.initializers.he_uniform(),
                kernel_regularizer=keras.regularizers.l2(self.args.kernel_regularizer),
                bias_regularizer=keras.regularizers.l2(self.args.bias_regularizer)
            )
        elif self.args.conv_type == "ds":
            self.conv = partial(
                keras.layers.SeparableConv2D,
                kernel_size=self.args.kernel_size,
                strides=self.args.stride,
                padding=self.args.padding,
                # # depth_multiplier=self.args.depth_multiplier,
                # activation=keras.layers.Activation(self.args.activation),
                # # use_bias=self.args.use_bias,
                # depthwise_initializer=keras.initializers.GlorotUniform(),
                # pointwise_initializer=keras.initializers.GlorotUniform(),
                # bias_initializer=keras.initializers.Zeros(),
                # depthwise_regularizer=keras.regularizers.l2(self.args.kernel_regularizer),
                # pointwise_regularizer=keras.regularizers.l2(self.args.kernel_regularizer),
                # bias_regularizer=keras.regularizers.l2(self.args.bias_regularizer),
                # activity_regularizer=keras.regularizers.l2(self.args.bias_regularizer),
                # # depthwise_constraint=self.args.depthwise_constraint,
                # # pointwise_constraint=self.args.pointwise_constraint,
                # # bias_constraint=self.args.bias_constraint
            )
    
        else:
            raise AttributeError("Non-valid conv_type specified")

        # Max pooling layer
        self.max_pooling = partial(
            keras.layers.MaxPooling2D,
            pool_size=(2,2),
            padding=self.args.padding
        )

        # Average pooling layer
        self.average_pooling = partial(
            keras.layers.AveragePooling2D,
            pool_size=(2,2),
            padding=self.args.padding
        )

        # Set default pooling based on provided args
        if self.args.stride > 1:
            # Don't pool, dimensionality reduction is done via convolutions
            self.pooling = keras.layers.Identity
        elif self.args.pooling == "average":
            self.pooling = self.average_pooling
        elif self.args.pooling == "max":
            self.pooling = self.max_pooling
        else:
            raise AttributeError("Non-valid pooling type: 'max' or 'average' are valid types")
        
        # Batch normalization layer
        self.batch_norm = partial(
            keras.layers.BatchNormalization,
            momentum=0.9
        )   
            
        self.mc_dropout = keras.layers.Dropout
        self.mc_spatial_dropout = keras.layers.SpatialDropout2D

        self.increase_filters = True if self.args.stride == 1 or self.args.conv_type == "depthwise_separable" else False

        self.build_model()

    def build_model(self):

        inputs = keras.Input(shape=[SKYRMION.H, SKYRMION.W, SKYRMION.C], dtype="float32")
        # hidden = keras.layers.Rescaling(scale=1 / 255)(inputs)            

        for i in range(self.args.depth):
            num_filters = self.args.filters << i if self.increase_filters else self.args.filters
            hidden = self.group_conv(inputs, filters=num_filters)
            hidden = self.mc_spatial_dropout(rate=self.args.dropout)(hidden)
        
        # hidden = keras.layers.Flatten()(hidden)
        hidden = keras.layers.GlobalAveragePooling2D()(hidden)
        
        hidden = self.group_dense(hidden, units=num_filters * 4)
        hidden = self.mc_dropout(rate=self.args.dropout * 2)(hidden)
        hidden = self.group_dense(hidden, units=num_filters) # I've added this because it doesn't make sense to me to use dropout just before output
        
        outputs = self.dense(units=len(SKYRMION.LABELS), activation="softmax")(hidden)

        super().__init__(inputs, outputs)

    def group_conv(self, input, filters):

        hidden = self.conv(filters=filters)(input)
        hidden = self.batch_norm()(hidden)
        hidden = keras.layers.Activation(self.args.activation)(hidden)
        output = self.pooling()(hidden)

        return output

    def group_dense(self, input, units):

        hidden = self.dense(units)(input)
        hidden = self.batch_norm()(hidden)
        output = keras.layers.Activation(self.args.activation)(hidden)

        return output

    def get_config(self):
            
            config = super().get_config()  
            config["args"] = vars(self.args)

            return config

    @classmethod
    def from_config(cls, config):

        args = config.pop("args") 

        return cls(argparse.Namespace(**args), **config)

class Model(keras.Model):
    def _activation(self, inputs, args: argparse.Namespace):
        """Apply activation function with a fallback to 'relu' if invalid."""
        try:
            activation_function = getattr(keras.activations, args.activation, keras.activations.relu)
            return keras.layers.Activation(activation=activation_function)(inputs)
        except AttributeError:
            raise ValueError(f"Invalid activation function '{args.activation}', using 'relu' instead. Check keras.activations for valid activations")


# I am not really testing this model at all (sofar)
class ResNet(Model):
    def _cnn(self, inputs, args, filters, kernel_size, stride, activation):
        hidden = keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
        hidden = keras.layers.BatchNormalization()(hidden)
        hidden = self._activation(hidden, args) if activation else hidden
        return hidden

    def _block(self, inputs, args, filters, stride, layer_index):
        hidden = self._cnn(inputs, args, filters, 3, stride, activation=True)
        hidden = self._cnn(hidden, args, filters, 3, 1, activation=False)
        if stride > 1:
            residual = self._cnn(inputs, args, filters, 1, stride, activation=False)
        else:
            residual = inputs
        hidden = residual + keras.layers.Dropout(args.stochastic_depth * layer_index, noise_shape=[None, 1, 1, 1])(hidden)
        hidden = self._activation(hidden, args)
        return hidden

    def __init__(self, args):
        n = (args.depth - 2) // 6

        inputs = keras.Input(shape=[SKYRMION.H, SKYRMION.W, SKYRMION.C], dtype="float32")
        hidden = keras.layers.Rescaling(scale=1 / 255)(inputs)
        hidden = self._cnn(hidden, args, 16, 3, 1, activation=True)
        for stage in range(3):
            for block in range(n):
                hidden = self._block(hidden, args, 16 * (1 << stage), 2 if stage > 0 and block == 0 else 1, (stage * n + block) / (3 * n - 1))
        hidden = keras.layers.GlobalAvgPool2D()(hidden)
        hidden = keras.layers.Dropout(args.dropout)(hidden)
        outputs = keras.layers.Dense(len(SKYRMION.LABELS), activation="softmax")(hidden)
        super().__init__(inputs, outputs)