import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from Proposed_model import Transformer_model  # Import your model 


DATASET_PATH = r"E:\MS_IITM\Conference_data_2025\For_paper2\Train_data_part1\Train_json_sample"  # Update to your .json files directory
CHECKPOINT_DIR = r"D:\MS_IITM\Conference paper\Conference 2025\MODELS\Conference_2\Transformer_encoder_CNN_decoder\Model4_Tansformer_encoder_CNN_decoder_with_model_save\checkpoints"
FINAL_MODEL_PATH = r"D:\MS_IITM\Conference paper\Conference 2025\MODELS\Conference_2\Transformer_encoder_CNN_decoder\Model4_Tansformer_encoder_CNN_decoder_with_model_save\final_mode\final_model.h5"


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list, dataset_path, batch_size, input_size, shuffle=True):
        self.file_list = file_list
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.file_list[index * self.batch_size:(index + 1) * self.batch_size]

        ppg_batch = []
        abp_batch = []

        for file_name in batch_files:
            file_path = os.path.join(self.dataset_path, file_name) 
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Ensure correct length and append
                if len(data['PPG_values']) == 1250 and len(data['ABP_values']) == 1250: 
                    ppg_batch.append(data['PPG_values'])
                    abp_batch.append(data['ABP_values'])


        ppg_batch = np.array(ppg_batch, dtype=np.float32)[..., np.newaxis]
        abp_batch = np.array(abp_batch, dtype=np.float32)[..., np.newaxis].reshape(self.batch_size, 1, 1250, 1)

        return ppg_batch, abp_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def mae_percentage(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()
    return tf.reduce_mean(tf.abs(y_true - y_pred) / (tf.abs(y_true) + epsilon)) * 100


class CustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint, checkpoint_prefix):
        super(CustomCheckpoint, self).__init__()
        self.checkpoint = checkpoint
        self.checkpoint_prefix = checkpoint_prefix
        self.best_val_mae = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val_mae = logs.get('val_mae')
        if current_val_mae is not None and current_val_mae < self.best_val_mae:
            print(f"\nImproved val_mae from {self.best_val_mae:.4f} to {current_val_mae:.4f}. Saving checkpoint.")
            self.best_val_mae = current_val_mae


            for filename in os.listdir(CHECKPOINT_DIR):
                if filename.startswith("ckpt"):  
                    file_path = os.path.join(CHECKPOINT_DIR, filename)
                    try:
                        os.remove(file_path)
                        print(f"Removed old checkpoint file: {filename}")
                    except Exception as e:
                        print(f"Error removing old checkpoint file: {filename} - {e}")

            self.checkpoint.save(file_prefix=self.checkpoint_prefix)









if __name__ == "__main__":
    print("Loading dataset...")
    
 
    file_list = [f for f in os.listdir(DATASET_PATH) if f.endswith('.json')]


    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)
    print(f"Training set size: {len(train_files)}")
    print(f"Validation set size: {len(val_files)}")

    train_generator = DataGenerator(train_files, DATASET_PATH, batch_size=16, input_size=1250)
    val_generator = DataGenerator(val_files, DATASET_PATH, batch_size=16, input_size=1250, shuffle=False)
    print("Building model...")
    input_signal = tf.keras.layers.Input(shape=(1250, 1), name="PPG_Input")
    model = Transformer_model(input_signal) 
    print(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"tf.reshape_1": "mae"},
        metrics=["mse", "mae", mae_percentage]
    )


    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")


    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Restoring from checkpoint:", latest_checkpoint)
        checkpoint.restore(latest_checkpoint)


        try:
            epoch_number = int(latest_checkpoint.split('-')[-1]) 
            initial_epoch = epoch_number + 1  # Start from the next epoch
        except:
            initial_epoch = 0  # If extraction fails, start from 0
    else:
        initial_epoch = 0

    checkpoint_callback = CustomCheckpoint(checkpoint, checkpoint_prefix)

 
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=1)

    # Train the model
    print("Training model...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=400,
        initial_epoch=initial_epoch,  # Pass the initial_epoch here
        callbacks=[early_stopping, reduce_lr_callback, checkpoint_callback]  # Removed best_model_checkpoint
    )

    # Save the final model 
    print("Saving final model...")
    model.save(FINAL_MODEL_PATH, overwrite=True)

    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_mse, test_mae_percentage = model.evaluate(val_generator)
    print(f"Validation Loss (MSE): {test_loss}")
    print(f"Validation MSE: {test_mse}")
    print(f"Validation MAE (%): {test_mae_percentage:.2f}%")
