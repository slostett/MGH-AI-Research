import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from python_tools.iotools import ResultsCache
from python_tools.export import export_fig
from python_tools.numerical import split_factors
from python_tools.dict import merge
from python_tools.deep_learning import write_tfrecord_v2, read_tfr

# For patching and image preprocessing
from python_tools.oper import OperPatchPartitioner
from python_tools.misc import extract_defaults

# Import setup and data tools
import gtv_segmentation_setup as gtv

def transformer_block(x, num_heads, key_dim):
    """
    Apply a Transformer block with multi-head self-attention and feedforward layers.

    Parameters:
        x (Tensor): Input tensor.
        num_heads (int): Number of attention heads.
        key_dim (int): Dimension of each attention head.

    Returns:
        Tensor: Output tensor after applying attention and feedforward network.
    """
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    ffn_output = layers.Dense(x.shape[-1], activation='relu')(x)
    ffn_output = layers.Dense(x.shape[-1])(ffn_output)
    x = layers.Add()([x, ffn_output])
    x = layers.LayerNormalization()(x)
    return x

def squeeze_excite_block(input_tensor, ratio=16):
    """
    Apply a squeeze-and-excitation block.

    Parameters:
        input_tensor (Tensor): Input tensor.
        ratio (int): Reduction ratio.

    Returns:
        Tensor: Output tensor after SE block.
    """
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape([1, 1, filters])(se)
    return layers.Multiply()([input_tensor, se])

def build_multimodal_unet(input_shape=(512, 512, 3), num_classes=1, base_filters=32, use_transformer=True):
    """
    Build a 2D U-Net model with optional Transformer blocks, skip connections, and SE blocks.

    Parameters:
        input_shape (tuple): Shape of the input tensor.
        num_classes (int): Number of output classes.
        base_filters (int): Number of filters to start with.
        use_transformer (bool): Whether to include Transformer blocks in bottleneck.

    Returns:
        tf.keras.Model: Compiled U-Net model.
    """
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = squeeze_excite_block(x)
        return x

    def encoder_block(x, filters):
        f = conv_block(x, filters)
        p = layers.MaxPooling2D((2, 2))(f)
        return f, p

    def decoder_block(x, skip, filters):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        return x

    inputs = keras.Input(shape=input_shape)
    s1, p1 = encoder_block(inputs, base_filters)
    s2, p2 = encoder_block(p1, base_filters * 2)
    s3, p3 = encoder_block(p2, base_filters * 4)
    s4, p4 = encoder_block(p3, base_filters * 8)

    b1 = conv_block(p4, base_filters * 16)

    if use_transformer:
        b1_shape = tf.keras.backend.int_shape(b1)
        flat_b1 = layers.Reshape((-1, b1_shape[-1]))(b1)
        t_block = transformer_block(flat_b1, num_heads=4, key_dim=64)
        b1 = layers.Reshape((b1_shape[1], b1_shape[2], b1_shape[3]))(t_block)

    d1 = decoder_block(b1, s4, base_filters * 8)
    d2 = decoder_block(d1, s3, base_filters * 4)
    d3 = decoder_block(d2, s2, base_filters * 2)
    d4 = decoder_block(d3, s1, base_filters)

    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="sigmoid")(d4)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def visualize_predictions(model, dataset, num_examples=3):
    """
    Visualize model predictions against ground truth on a sample of images.

    Parameters:
        model (tf.keras.Model): Trained model to use for prediction.
        dataset (tf.data.Dataset): Dataset of image-label pairs.
        num_examples (int): Number of examples to visualize.
    """
    for i, (img, lbl) in enumerate(dataset.take(num_examples)):
        pred = model.predict(tf.expand_dims(img, 0))[0, ..., 0]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img[..., 0], cmap='gray')
        axes[0].set_title('CT Image')
        axes[1].imshow(lbl[..., 0], cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[2].imshow(pred, cmap='hot')
        axes[2].set_title('Prediction')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

def plot_training_history(history):
    """
    Plot training and validation loss and accuracy over time.

    Parameters:
        history (keras.callbacks.History): Training history returned by model.fit().
    """
    metrics = ['loss', 'accuracy']
    fig, axs = plt.subplots(1, len(metrics), figsize=(12, 4))
    for i, metric in enumerate(metrics):
        axs[i].plot(history.history[metric], label='train')
        axs[i].plot(history.history['val_' + metric], label='val')
        axs[i].set_title(metric.capitalize())
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
    plt.tight_layout()
    plt.show()
    # Save figure using export_fig from gtv setup
    export_fig(fig, size=[12, 4], dpi=300,
               output_folder='.', output_fname='training_metrics')

def train_and_evaluate_unet(input_shape=(512, 512, 3), batch_size=8, num_epochs=20, model_dir='./gtv_model/'):
    """
    Train the multimodal U-Net model and evaluate on a validation dataset.

    Parameters:
        input_shape (tuple): Shape of input images.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        model_dir (str): Path to save model and training history.

    Returns:
        model (tf.keras.Model): Trained U-Net model.
        history (History): Keras training history object.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Prepare TFRecords using setup functions
    data = gtv.run_exp_prep_data(
        gtv.data_fold,
        patient_idx_tr=['p2', 'p3', 'p4'],
        patient_idx_tst=['p5'],
        modality_list=['ct', 'mr', 'pt'],
        flag_comb=True,
        flag_mask_only=True,
        patch_params={'patch_size': [32, 32], 'patch_overlap': [16, 16]},
        flag_skip_write=False
    )

    train_tfrecords = data['fname_info']['tfr_fname_list_tr']
    val_tfrecords = list(data['fname_info']['tfr_fname_list_tst'].values())

    model = build_multimodal_unet(input_shape=input_shape, use_transformer=True)

    def _parse_example(example_proto):
        features = {'image': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.string)}
        parsed = tf.io.parse_single_example(example_proto, features)
        image = tf.io.decode_raw(parsed['image'], tf.float32)
        label = tf.io.decode_raw(parsed['label'], tf.float32)
        image = tf.reshape(image, input_shape)
        label = tf.reshape(label, input_shape[:2] + (1,))
        return image, label

    train_dataset = tf.data.TFRecordDataset(train_tfrecords)
    train_dataset = train_dataset.map(_parse_example)
    train_dataset = train_dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.TFRecordDataset(val_tfrecords)
    val_dataset = val_dataset.map(_parse_example)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Model training
    history = model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)

    # Save model and history
    model.save(os.path.join(model_dir, 'unet_gtv_model.h5'))
    with open(os.path.join(model_dir, 'train_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    print("\nVisualization of predictions on validation set:")
    visualize_predictions(model, val_dataset)

    print("\nPlotting training metrics:")
    plot_training_history(history)

    return model, history

if __name__ == "__main__":
    model, history = train_and_evaluate_unet()