import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Input, Reshape, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np

class WeightMultiplicationLayer(tf.keras.layers.Layer):
    def __init__(self, multiplication_weights, **kwargs):
        super(WeightMultiplicationLayer, self).__init__(**kwargs)
        self.multiplication_weights = multiplication_weights

    def call(self, inputs):
        return inputs * self.multiplication_weights  # Element-wise multiplication

    def get_config(self):
        config = super().get_config()
        config.update({
            "multiplication_weights": self.multiplication_weights.numpy()
        })
        return config
    

def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] * np.arange(d_model)[np.newaxis, :] / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) 
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  
    return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)



def transformer_encoder(input_tensor, d_model, num_heads, ff_dim, dropout_rate=0.1):
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(input_tensor, input_tensor)
    attn_output = Dropout(dropout_rate)(attn_output)

    out1 = LayerNormalization(epsilon=1e-6)(input_tensor + attn_output)
    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(d_model)(ffn)
    ffn_output = Dropout(dropout_rate)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return out2

# CNN Decoder
'''
def cnn_decoder(input_tensor):

    flattened = Flatten()(input_tensor)
    dense1 = Dense(512, activation='relu')(flattened)
    dense2 = Dense(256, activation='relu')(dense1)
    output = Dense(1250, activation='linear')(dense2)
    reshaped_output = Reshape((1, 1250, 1))(output)
    return reshaped_output
'''
def cnn_decoder(input_tensor):
    #print("\n\n\n\n\n\n\n")
    #print("Input Tensor:", input_tensor.shape)
    
    # Apply a Conv2D layer with a kernel size of (2, 2)
    conv1 = Conv2D(64, (5, 5), activation='relu', padding='same')(input_tensor)
    #print("conv1:", conv1.shape)
    
    # Apply MaxPooling2D with pool size (1, 2) to keep the first dimension and reduce the second
    pool1 = MaxPooling2D(pool_size=(1, 2))(conv1)

    # Apply a Conv2D layer with a kernel size of (2, 2)
    #conv2 = Conv2D(128, (5, 5), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (1, 5), activation='relu', padding='valid')(pool1) 
    #pool2 = MaxPooling2D(pool_size=(1, 3))(conv2)

    conv3 = Conv2D(1, (5, 5), activation='relu', padding='same')(conv2)
    conv3_reshape = tf.reshape(conv3, (-1, 1, 1250, 1))


    return conv3_reshape

'''

def cnn_decoder(input_tensor):
    # Apply a Conv2D layer with a kernel size of (3, 3)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(1, 2))(conv1)

    # Apply a Conv2D layer with a kernel size of (3, 3)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    #pool2 = MaxPooling2D(pool_size=(1, 3))(conv2)  # Remove this line

    # Conv2D for downsampling (replace MaxPooling2D)
    conv3 = Conv2D(128, (1, 5), activation='relu', padding='valid')(conv2) 

    # Apply a Conv2D layer with a kernel size of (5, 5)
    conv4 = Conv2D(1, (5, 1), activation='relu', padding='same')(conv3)
    conv4_reshape = tf.reshape(conv4, (-1, 1, 1250, 1))

    return conv4_reshape
'''
# Transformer Model
'''
d_model  [default: 512]
d_ff  [default: 2048]
n_head  [default: 8]
Batch size  [default: 128]
'''




def Transformer_model(input_signal):
    # Model parameters
    input_length = 1250
    embedding_size = 125
    num_embeddings = input_length // embedding_size  
    num_encoders = 10
    num_heads = 8
    d_model = 125
    ff_dim = 512
    
    print(f"\n\n\n\n Type of input signal: {type(input_signal)}")

    # Step 1: Reshape input to (None, num_embeddings, embedding_size)
    embeddings = Reshape((num_embeddings, embedding_size))(input_signal)
    print(f"Shape of embeddings after reshape: {embeddings.shape}")

    pos_encoding = positional_encoding(num_embeddings, embedding_size)
    embeddings += pos_encoding

    # Step 2: Define trainable weights for each encoder block
    multiplication_weights = [tf.Variable(1.0, trainable=True, dtype=tf.float32) for _ in range(num_encoders)]

    # Step 3: Pass through Transformer Encoder layers
    encoder_outputs = []
    current_output = embeddings
    for i in range(num_encoders):
        current_output = transformer_encoder(current_output, d_model, num_heads, ff_dim)
        # Multiply each encoder output by its corresponding weight
        current_output = WeightMultiplicationLayer(multiplication_weights[i])(current_output)
        print(f"Shape of current_output after encoder {i + 1}: {current_output.shape}")
        encoder_outputs.append(current_output)

    # Step 4: Stack encoder outputs (Shape: (None, 10, 125, 20))
    stacked_encoder_outputs = tf.stack(encoder_outputs, axis=-1)
    print(f"Shape of stacked_encoder_outputs: {stacked_encoder_outputs.shape}")

    # Step 5: Reshape stacked encoder outputs from (None, 10, 125, 10) to (None, 1250, 10)
    reshaped_output = tf.reshape(stacked_encoder_outputs, (-1, 1250, num_encoders))  # shape: (None, 1250, 10)
    reshaped_output = tf.expand_dims(reshaped_output, -1)  # shape: (None, 1250, 10, 1)
    print(f"Shape after reshaping: {reshaped_output.shape}")


    # Step 6: Apply CNN Decoder for ABP prediction
    abp_output = cnn_decoder(reshaped_output)
    print(f"Shape of abp_output after CNN decoder: {abp_output.shape}")
 

    print(f"\n\n\n\n Type of final_output_abp: {type(abp_output)}") 
    # Define the model
    model = Model(inputs=input_signal, outputs=abp_output, name="Transformer_Encoder_CNN_Decoder")

    model.summary() 
    return model
