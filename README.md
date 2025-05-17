# Transformer_CNN_Predicting_ABP_from_PPG
The input to the model is PPG signal with of 10 sec duration with 125Hz sampling rate.
Output obtained will be ABP signal of 10 sec duration.

- Transformer based Encoder -> For converting PPG signals to PPG features
- Convolution based Decoder -> For predicting ABP signal from the derived PPG features

The PPG signal is divided into 10 segment each containing data of 1 sec, the segment acts as embeddings to the Transformer model. Also a postional encoding using cos-sine is used for giving position for as the input.
  
![Model_2](https://github.com/user-attachments/assets/8acf2ca5-de3e-4c88-98f6-67bdef3be36e)
