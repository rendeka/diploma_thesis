import keras
from skyrmion_dataset import SKYRMION

class Model(keras.Model):
    def _activation(self, inputs, args):
        valid_activations = [func for func in dir(keras.activations) if not func.startswith("__")]
        if args.activation in valid_activations:
            activation_function = args.activation
        else:
            activation_function = "relu"
            raise ValueError(f"'{args.activation}' is not a valid activation function 'relu' will be used. See 'keras.activations' for valid activation funcitons")

        return keras.layers.Activation(activation=activation_function)(inputs)


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


class Model5(Model):
    def __init__(self, args):

        from functools import partial
        super().__init__()

        self._Dense = partial(
            keras.layers.Dense,
            activation=None,
            use_bias=True, 
            kernel_initializer="glorot_normal", 
            bias_initializer="glorot_uniform", 
            kernel_regularizer=keras.regularizers.l2(1.e-4), 
            bias_regularizer=keras.regularizers.l2(1.e-5)
        )

        # 2D convolutional layer
        self._Conv2D = partial(
            keras.layers.Conv2D,
            kernel_size=(3,3),
            activation=None,
            padding="same",
            kernel_initializer="he_normal", 
            bias_initializer="he_uniform",
            kernel_regularizer=keras.regularizers.l2(1.e-4),
            bias_regularizer=keras.regularizers.l2(1.e-5)
        )

        # 2D convolutional pooling layer
        self._ConvPooling2D = partial(
            keras.layers.Conv2D,
            kernel_size=(2,2),
            strides=(2,2),
            activation=None,
            padding="valid",
            kernel_initializer="he_normal", 
            bias_initializer="he_uniform",
            kernel_regularizer=keras.regularizers.l2(1.e-4),
            bias_regularizer=keras.regularizers.l2(1.e-5)
        )

        # 2D max pooling layer
        self._MaxPooling2D = partial(
            keras.layers.MaxPooling2D,
            pool_size=(2,2),
            padding="same"
        )

        # 2D average pooling layer
        self._AveragePooling2D = partial(
            keras.layers.AveragePooling2D,
            pool_size=(2,2),
            padding="same"
        )

        # batch normalization layer
        self._BatchNormalization = partial(
            keras.layers.BatchNormalization,
            momentum=0.9
        )

        # Monte Carlo Dropout layer
        class MCDropout(keras.layers.Dropout):
            """Monte Carlo Dropout layer."""
            
            def call(self, inputs):
                return super().call(inputs, training=True)

        # Monte Carlo 2D spatial dropout layer
        class MCSpatialDropout2D(keras.layers.SpatialDropout2D):
            """Monte Carlo 2D spatial Dropout layer."""
            
            def call(self, inputs):
                return super().call(inputs, training=True)
            
        self._MCDropout = MCDropout
        self._MCSpatialDropout2D = MCSpatialDropout2D

        self._GroupConv2D = partial(
            self.GroupConv2D,
            Conv2D=self._Conv2D,
            BatchNormalization=self._BatchNormalization,
            Activation=keras.layers.Activation,
            AveragePooling2D=self._AveragePooling2D
        )

        self._GroupDense = partial(
            self.GroupDense,
            Dense=self._Dense,
            BatchNormalization=self._BatchNormalization,
            Activation=keras.layers.Activation
        )

        inputs = keras.Input(shape=[SKYRMION.H, SKYRMION.W, SKYRMION.C], dtype="float32")
        # hidden = keras.layers.Rescaling(scale=1 / 255)(inputs)            
        # feature extraction
        hidden = self._GroupConv2D(filters=8)(inputs)
        hidden = self._MCSpatialDropout2D(rate=0.1)(hidden)
        
        hidden = self._GroupConv2D(filters=16)(hidden)
        hidden = self._MCSpatialDropout2D(rate=0.2)(hidden)
        
        hidden = self._GroupConv2D(filters=32)(hidden)
        hidden = self._MCSpatialDropout2D(rate=0.2)(hidden)
        
        hidden = self._GroupConv2D(filters=64)(hidden)
        hidden = self._MCSpatialDropout2D(rate=0.2)(hidden)

        hidden = self._Conv2D(filters=32)(hidden)
        hidden = self._BatchNormalization()(hidden)
        # hidden = keras.layers.Activation("selu")(hidden)
        hidden = self._activation(hidden, args)
        
        # classification
        hidden = keras.layers.Flatten()(hidden)
        
        hidden = self._GroupDense(units=128)(hidden)
        hidden = self._MCDropout(rate=0.2)(hidden)
        
        outputs = self._Dense(units=len(SKYRMION.LABELS), activation="softmax")(hidden)

        super().__init__(inputs, outputs)

    def GroupConv2D(self, filters, Conv2D, BatchNormalization, Activation, AveragePooling2D):
        """Conv2D -> BatchNormalization -> SELU -> Average pooling."""
        
        def Layers(x):
            # convolutional layer
            x = Conv2D(filters=filters)(x)
            # batch normalization
            x = BatchNormalization()(x)
            # activation
            x = Activation(activation="selu")(x)
            # average poRandom seed.oling
            x = AveragePooling2D()(x)
                    
            return x
        
        return Layers

    def GroupDense(self, units, Dense, BatchNormalization, Activation):
        """Dense -> BatchNormalization -> ELU."""
        
        def Layers(x):
            # dense layer
            x = Dense(units)(x)
            # batch normalization
            x = BatchNormalization()(x)
            # activation
            x = Activation(activation="elu")(x)
                    
            return x
        
        return Layers


# class Model5():
#     def __init__(self, args):

#         from functools import partial
#         super().__init__()

#         self._Dense = partial(
#             keras.layers.Dense,
#             activation=None,
#             use_bias=True, 
#             kernel_initializer="glorot_normal", 
#             bias_initializer="glorot_uniform", 
#             kernel_regularizer=keras.regularizers.l2(1.e-4), 
#             bias_regularizer=keras.regularizers.l2(1.e-5)
#         )

#         # 2D convolutional layer
#         self._Conv2D = partial(
#             keras.layers.Conv2D,
#             kernel_size=(3,3),
#             activation=None,
#             padding="same",
#             kernel_initializer="he_normal", 
#             bias_initializer="he_uniform",
#             kernel_regularizer=keras.regularizers.l2(1.e-4),
#             bias_regularizer=keras.regularizers.l2(1.e-5)
#         )

#         # 2D convolutional pooling layer
#         self._ConvPooling2D = partial(
#             keras.layers.Conv2D,
#             kernel_size=(2,2),
#             strides=(2,2),
#             activation=None,
#             padding="valid",
#             kernel_initializer="he_normal", 
#             bias_initializer="he_uniform",
#             kernel_regularizer=keras.regularizers.l2(1.e-4),
#             bias_regularizer=keras.regularizers.l2(1.e-5)
#         )

#         # 2D max pooling layer
#         self._MaxPooling2D = partial(
#             keras.layers.MaxPooling2D,
#             pool_size=(2,2),
#             padding="same"
#         )

#         # 2D average pooling layer
#         self._AveragePooling2D = partial(
#             keras.layers.AveragePooling2D,
#             pool_size=(2,2),
#             padding="same"
#         )

#         # batch normalization layer
#         self._BatchNormalization = partial(
#             keras.layers.BatchNormalization,
#             momentum=0.9
#         )

#         def GroupConv2D(self, filters):
#             """Conv2D -> BatchNormalization -> SELU -> Average pooling."""
            
#             def Layers(x):
#                 # convolutional layer
#                 x = self._Conv2D(filters=filters)(x)
#                 # batch normalization
#                 x = self._BatchNormalization()(x)
#                 # activation
#                 x = keras.layers.Activation(activation="selu")(x)
#                 # average pooling
#                 x = self._AveragePooling2D()(x)
                        
#                 return x
            
#             return Layers

#         ###################################################################################################

#         def GroupDense(self, units):
#             """Dense -> BatchNormalization -> ELU."""
            
#             def Layers(x):
#                 # dense layer
#                 x = self._Dense(units)(x)
#                 # batch normalization
#                 x = self._BatchNormalization()(x)
#                 # activation
#                 x = keras.layers.Activation(activation="elu")(x)
                        
#                 return x
            
#             return Layers

#         inputs = keras.Input(shape=[SKYRMION.H, SKYRMION.W, SKYRMION.C], dtype="float32")
#         hidden = keras.layers.Rescaling(scale=1 / 255)(inputs)            
#         # feature extraction
#         hidden = self._GroupConv2D(filters=8)(inputs)
#         hidden = self._MCSpatialDropout2D(rate=0.1)(hidden)
        
#         hidden = self._GroupConv2D(filters=16)(hidden)
#         hidden = self._MCSpatialDropout2D(rate=0.2)(hidden)
        
#         hidden = self._GroupConv2D(filters=32)(hidden)
#         hidden = self._MCSpatialDropout2D(rate=0.2)(hidden)
        
#         hidden = self._GroupConv2D(filters=64)(hidden)
#         hidden = self._MCSpatialDropout2D(rate=0.2)(hidden)

#         hidden = self._Conv2D(filters=32)(hidden)
#         hidden = self._BatchNormalization()(hidden)
#         hidden = keras.layers.Activation("selu")(hidden)
        
#         # classification
#         hidden = keras.layers.Flatten()(hidden)
        
#         hidden = self._GroupDense(units=128)(hidden)
#         hidden = self._MCDropout(rate=0.2)(hidden)
        
#         outputs = self._Dense(units=len(SKYRMION.LABELS), activation="softmax")(hidden)

#         super().__init__(inputs, outputs)