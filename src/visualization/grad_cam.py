import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
import matplotlib.pyplot as plt
import mlflow

def load_sample_image():
    """Load a sample image for Grad-CAM"""
    X_test = np.load('data/processed/X_test.npy')
    return X_test[0]  # Use first test image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Generate Grad-CAM heatmap for a CNN model"""
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1) / 255.0

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    """Save Grad-CAM visualization"""
    heatmap = np.uint8(255 * heatmap)

    colormap = plt.colormaps.get_cmap("jet")
    jet_colors = colormap(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Convert grayscale to RGB
    if img.ndim == 2:
        img_rgb = np.stack([img]*3, axis=-1)
    else:
        img_rgb = img

    superimposed_img = jet_heatmap * alpha + img_rgb
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    plt.figure()
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.savefig('reports/grad_cam.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def log_gradcam_to_mlflow():
    """Log Grad-CAM visualization to MLflow"""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Dyslexia_Detection_HPO")

    with mlflow.start_run():
        mlflow.log_artifact('reports/grad_cam.png')
        print("✅ Grad-CAM image logged to MLflow.")

if __name__ == '__main__':
    # Load sample image
    img = load_sample_image()

    # Load trained CNN model
    model = tf.keras.models.load_model('models/feature_extractor.h5')

    # Define last conv layer (adjust this based on your model)
    last_conv_layer_name = 'conv2d_3'  # change if your model uses a different name

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)

    # Save Grad-CAM image
    save_and_display_gradcam(img, heatmap)

    # Log to MLflow
    log_gradcam_to_mlflow()
