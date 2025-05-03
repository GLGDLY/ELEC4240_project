import tensorflow as tf
from tensorflow.keras import layers, Model

class UNet(Model):
    def __init__(self, input_size=(256, 256, 3)):
        super(UNet, self).__init__()
        
        # Encoder
        def encoder_block(x, filters, kernel_size=3, padding='same', strides=1):
            x = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, 
                            kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            x = layers.Conv2D(filters, kernel_size, padding=padding, strides=1,
                            kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            return x

        # Decoder
        def decoder_block(x, skip_features, filters, kernel_size=3, padding='same', strides=1):
            x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
            
            x = layers.Concatenate()([x, skip_features])
            
            x = layers.Conv2D(filters, kernel_size, padding=padding,
                            kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            x = layers.Conv2D(filters, kernel_size, padding=padding,
                            kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            return x

        inputs = layers.Input(input_size)

        # Encoder Path
        e1 = encoder_block(inputs, 64)
        p1 = layers.MaxPooling2D((2, 2))(e1)

        e2 = encoder_block(p1, 128)
        p2 = layers.MaxPooling2D((2, 2))(e2)

        e3 = encoder_block(p2, 256)
        p3 = layers.MaxPooling2D((2, 2))(e3)

        e4 = encoder_block(p3, 512)
        p4 = layers.MaxPooling2D((2, 2))(e4)

        b1 = encoder_block(p4, 1024)

        # Decoder Path
        d1 = decoder_block(b1, e4, 512)

        d2 = decoder_block(d1, e3, 256)

        d3 = decoder_block(d2, e2, 128)

        d4 = decoder_block(d3, e1, 64)

        outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d4)

        self.model = Model(inputs, outputs, name="U-Net")

    def call(self, inputs):
        return self.model(inputs)
