#!/usr/bin/env python3
import keras
import argparse
from functools import partial
from skyrmion_dataset import SKYRMION

@keras.saving.register_keras_serializable()
class ModelBase(keras.Model):
    def __init__(self, args, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs)
        self.args = args

    class MCDropout(keras.layers.Dropout):
        def __init__(self, rate, mc_inference=False, **kwargs):
            super().__init__(rate, **kwargs)
            self.mc_inference = mc_inference

        def call(self, inputs, training=False):
            return super().call(inputs, training=(training or self.mc_inference))
        
    class MCSpatialDropout(keras.layers.SpatialDropout2D):
        def __init__(self, rate, mc_inference=False, **kwargs):
            super().__init__(rate, **kwargs)
            self.mc_inference = mc_inference

        def call(self, inputs, training=False):
            return super().call(inputs, training=(training or self.mc_inference))
        
    class MCAplhaDropout(keras.layers.AlphaDropout):
        def __init__(self, rate, mc_inference=False, **kwargs):
            super().__init__(rate, **kwargs)
            self.mc_inference = mc_inference

        def call(self, inputs, training = False):
            return super().call(inputs, training=(training or self.mc_inference))

    # Get Input shape
    @property
    def input_shape(self):
        return [SKYRMION.H, SKYRMION.W, SKYRMION.C]
    
    # Get number of output classes
    @property
    def num_classes(self):
        return len(SKYRMION.LABELS)
    
    # Get filter scaling factor (definning how the number of filters increase in the conv layers)
    @property
    def get_filter_scaling_factor(self):
        if "Flatten" in self.args.fag:
            return 1
        elif self.args.pooling == "max" or self.args.conv_type == "depthwise_separable":
            return 2
        elif self.args.stride > 1:
            return 1
        else:
            return 1.5
    
    # Activation
    @property
    def activation(self):
        return partial(
            keras.layers.Activation,
            self.args.activation
        )

    # Dense layer
    @property
    def dense(self):
        return partial(
            keras.layers.Dense,
            activation=self.activation(),
            use_bias=True, 
            kernel_initializer=keras.initializers.glorot_normal(), 
            bias_initializer=keras.initializers.glorot_uniform(), 
            kernel_regularizer=keras.regularizers.l2(self.args.kernel_regularizer), 
            bias_regularizer=keras.regularizers.l2(self.args.bias_regularizer)
        )

    # Convolutional layer
    @property
    def conv(self):
        if self.args.conv_type == "standard":
            return partial(
                keras.layers.Conv2D,
                kernel_size=self.args.kernel_size,
                activation=self.activation(),
                strides=self.args.stride,
                padding=self.args.padding,
                kernel_initializer=keras.initializers.he_normal(), 
                bias_initializer=keras.initializers.he_uniform(),
                kernel_regularizer=keras.regularizers.l2(self.args.kernel_regularizer),
                bias_regularizer=keras.regularizers.l2(self.args.bias_regularizer)
                )
        elif self.args.conv_type == "ds":
            return partial(
                keras.layers.SeparableConv2D,
                kernel_size=self.args.kernel_size,
                strides=self.args.stride,
                padding=self.args.padding,
            )
        else:
            raise AttributeError("Non-valid conv_type specified")

    # Max pooling layer
    @property
    def max_pooling(self):
        return partial(
            keras.layers.MaxPooling2D,
            pool_size=(2,2),
            padding=self.args.padding
        )

    # Average pooling layer
    @property
    def average_pooling(self):
        return partial(
            keras.layers.AveragePooling2D,
            pool_size=(2,2),
            padding=self.args.padding
        )

    # Set default pooling based on provided args
    @property
    def pooling(self):
        if self.args.stride > 1 or self.args.pooling == "no":
            # Don't pool, dimensionality reduction is done via convolutions
            return keras.layers.Identity
        elif self.args.pooling == "average":
            return self.average_pooling
        elif self.args.pooling == "max":
            return self.max_pooling
        else:
            raise AttributeError("Non-valid pooling type: 'max' or 'average' are valid types")
    
    # Batch normalization layer
    @property
    def batch_norm(self):
        if self.args.batch_norm:
            return partial(
                keras.layers.BatchNormalization,
                momentum=0.9
            ) 
        else:
            return keras.layers.Identity  
    
    def mc_dropout(self, rate):
        if self.args.activation == "selu" and self.args.alpha_dropout:
            return ModelBase.MCAplhaDropout(rate = rate)
        else:
            return ModelBase.MCDropout(rate = rate)
    
    def mc_spatial_dropout(self, rate):
        return ModelBase.MCSpatialDropout(rate = rate)

    def set_mc_inference(self, mc_inference: bool = False, rate=None):
        """Enable or disable MC Dropout after loading the model."""
        if rate:
            self.args.dropout = rate
        for layer in self.layers:
            if isinstance(layer, (ModelBase.MCDropout, ModelBase.MCSpatialDropout)):
                layer.mc_inference = mc_inference
        
    def conv_block(self, inputs, filters):

        hidden = self.conv(filters=filters)(inputs)
        hidden = self.batch_norm()(hidden)
        hidden = self.activation()(hidden)
        output = self.pooling()(hidden)

        return output

    def dense_block(self, inputs, units):

        hidden = self.dense(units)(inputs)
        hidden = self.batch_norm()(hidden)
        output = self.activation()(hidden)

        return output

    def head(self, inputs):
        if self.args.head == "softmax":
            outputs = self.dense(units=self.num_classes, activation="softmax")(inputs)
        else:
            hidden = self.dense(units=self.num_classes, activation=self.args.head)(inputs)
            outputs = hidden / keras.ops.sum(hidden, axis=1, keepdims=True)

        return outputs     
    
    def cbam_block(self, inputs, ratio=8):
        filters = inputs.shape[-1]

        # Channel Attention
        pool_average = keras.layers.GlobalAveragePooling2D()(inputs)
        pool_max = keras.layers.GlobalMaxPooling2D()(inputs)

        mlp_1 = keras.layers.Dense(filters // ratio, activation="relu")
        mlp_2 = keras.layers.Dense(filters, activation="sigmoid")

        output_avg = mlp_2(mlp_1(pool_average))
        output_max = mlp_2(mlp_1(pool_max))

        channel_attention = keras.layers.Add()([output_avg, output_max])
        channel_attention = keras.layers.Reshape((1, 1, filters))(channel_attention)

        hidden = keras.layers.Multiply()([inputs, channel_attention])

        # Spatial Attention
        pool_average = keras.layers.Lambda(
            lambda x: keras.ops.mean(x, axis=-1, keepdims=True), 
            output_shape=(hidden.shape[1], hidden.shape[2], 1))(hidden)
        pool_max = keras.layers.Lambda(
            lambda x: keras.ops.max(x, axis=-1, keepdims=True),
            output_shape=(hidden.shape[1], hidden.shape[2], 1))(hidden)

        concat = keras.layers.Concatenate(axis=-1)([pool_average, pool_max])
        spatial_attention = keras.layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
        outputs = keras.layers.Multiply()([hidden, spatial_attention])

        return outputs
    
    def se_block(self, inputs, ratio=8):
        """Squeeze-and-Excitation Block"""
        filters = inputs.shape[-1]
        
        se = keras.layers.GlobalAveragePooling2D()(inputs)
        se = keras.layers.Dense(filters // ratio, activation="relu")(se)
        se = keras.layers.Dense(filters, activation="sigmoid")(se)
        
        se = keras.layers.Reshape((1, 1, filters))(se)
        return keras.layers.Multiply()([inputs, se])
    
    def feature_aggregation(self, inputs):
        if "SE" in self.args.fag:
            inputs = self.se_block(inputs)
            return keras.layers.GlobalAveragePooling2D()(inputs)    
            
        if "GAP" in self.args.fag:
            return keras.layers.GlobalAveragePooling2D()(inputs)
        
        elif "Flatten" in self.args.fag:
            return keras.layers.Flatten()(inputs)

    def get_config(self):
            """Serialize the model configuration."""
            config = super().get_config()
            config["args"] = vars(self.args)  # Convert Namespace to dict
            return config

    @classmethod
    def from_config(cls, config):
        """Deserialize the model."""
        args = argparse.Namespace(**config.pop("args"))  # Convert dict back to Namespace
        return cls(args=args, **config)

class Model5(ModelBase):
    def __init__(self, args, **kwargs):
        self.args = args 
        self.filter_scaling_factor = self.get_filter_scaling_factor
        self.build_model(**kwargs)

    def build_model(self, **kwargs):
        inputs = keras.Input(shape=self.input_shape, dtype="float32")
        hidden = keras.layers.Rescaling(scale=1.0)(inputs)

        for i in range(self.args.depth):
            num_filters = int(self.args.filters * (self.filter_scaling_factor ** i))
            hidden = self.conv_block(hidden, filters=num_filters)
            hidden = self.mc_spatial_dropout(rate=self.args.spatial_dropout)(hidden)

        hidden = self.feature_aggregation(hidden)
        hidden = self.dense_block(hidden, units=num_filters * 4)
        hidden = self.mc_dropout(rate=self.args.dropout * 2)(hidden)
        hidden = self.dense_block(hidden, units=num_filters)
        outputs = self.head(hidden)

        super().__init__(args=self.args, inputs=inputs, outputs=outputs, **kwargs)

class ModelCBAM(ModelBase):
    def __init__(self, args, **kwargs):
        self.args = args 
        self.filter_scaling_factor = self.get_filter_scaling_factor
        self.build_model(**kwargs)

    def build_model(self, **kwargs):
        inputs = keras.Input(shape=self.input_shape)

        num_filters = self.args.filters
        hidden = self.conv(num_filters)(inputs)
        hidden = self.pooling()(hidden)

        for i in range(self.args.depth):
            num_filters = int(self.args.filters * (self.filter_scaling_factor ** i))
            hidden = self.residual_block_cbam(hidden, num_filters)
            hidden = self.pooling()(hidden)
        
        # hidden = keras.layers.GlobalAveragePooling2D()(hidden)
        # hidden = keras.layers.Dense(num_filters, activation="relu")(hidden)

        hidden = self.feature_aggregation(hidden)
        # hidden = self.dense_block(hidden, units=num_filters)
        outputs = self.head(hidden)
        
        super().__init__(args=self.args, inputs=inputs, outputs=outputs, **kwargs)

    def residual_block_cbam(self, inputs, filters):
        if inputs.shape[-1] == filters:
            skip = inputs
        else:
            skip = keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, activation=None)(inputs)

        hidden = self.conv(filters=filters)(inputs) 
        hidden = self.batch_norm()(hidden)
        # hidden = keras.layers.Activation("relu")(hidden)
        hidden = self.activation()(hidden)
        hidden = self.conv(filters=filters)(hidden)
        hidden = self.batch_norm()(hidden)
        hidden = self.cbam_block(hidden)
        hidden = keras.layers.Add()([hidden, skip])
        # outputs = keras.layers.Activation("relu")(hidden)
        outputs = self.activation()(hidden)

        return outputs
    
class ModelFFN(ModelBase):

    def __init__(self, args, **kwargs):
        self.args = args
        self.build_model(**kwargs)

    def build_model(self, **kwargs):
        inputs = keras.Input(shape=self.input_shape)
        hidden = keras.layers.Flatten()(inputs)
        hidden = keras.layers.Dense(units=self.args.filters)(hidden)
        hidden = self.activation()(hidden)
        outputs = self.head(hidden)

        super().__init__(args=self.args, inputs=inputs, outputs=outputs, **kwargs)

# I am not really testing this model at all (sofar)
class ResNet(ModelBase):
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