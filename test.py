import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2  # Import OpenCV

# Load the trained model
model_path = 'unet_breast_segmentation.keras'
model = load_model(model_path)

# Load data function
def load_data(image_dir, img_size=(224, 224)):
    images = []
    image_names = []
    for img_name in os.listdir(image_dir):
        img = tf.keras.preprocessing.image.load_img(os.path.join(image_dir, img_name), target_size=img_size, color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0  # Normalize to [0, 1]
        images.append(img)
        image_names.append(img_name)
    
    return np.array(images), image_names

# Paths to your test image directory
test_image_dir = 'abnormal'  # Adjust this path to your test images directory

# Load the test data
X_test, test_image_names = load_data(test_image_dir)

# Run predictions
predictions = model.predict(X_test)

# Threshold the predictions to get binary masks
threshold = 0.5
predictions = (predictions > threshold).astype(np.uint8)

# Save the predicted masks
output_dir = 'predicted_masks'
os.makedirs(output_dir, exist_ok=True)
for i, pred in enumerate(predictions):
    pred = (pred * 255).astype(np.uint8)  # Convert to 0-255 range for saving
    pred_image = tf.keras.preprocessing.image.array_to_img(pred)
    pred_image.save(os.path.join(output_dir, f'pred_{test_image_names[i]}'))

# Function to apply masks to the original images and save segmented parts
def save_segmented_parts(images, masks, original_image_names, save_dir='segmented_parts'):
    os.makedirs(save_dir, exist_ok=True)
    for img, mask, img_name in zip(images, masks, original_image_names):
        img = (img * 255).astype(np.uint8)  # Convert to original range
        img = img.squeeze()  # Remove extra dimension
        mask = mask.squeeze()  # Remove extra dimension
        segmented_part = cv2.bitwise_and(img, img, mask=mask)  # Apply mask to image
        segmented_part = np.expand_dims(segmented_part, axis=-1)
        segmented_img = tf.keras.preprocessing.image.array_to_img(segmented_part)
        segmented_img.save(os.path.join(save_dir, f'seg_{img_name}'))

# Apply the masks to the original images and save segmented parts
save_segmented_parts(X_test, predictions, test_image_names)

# Overlay masks on original images for visualization
def overlay_masks_on_images(images, masks, alpha=0.5):
    overlayed_images = []
    for img, mask in zip(images, masks):
        img = img.squeeze()  # Remove the extra dimension
        mask = mask.squeeze()  # Remove the extra dimension
        overlay = np.where(mask == 1, img * 0.5 + 0.5, img)  # Overlay the mask on the image
        overlayed_images.append(overlay)
    return overlayed_images

# Overlay the masks on the original images
overlayed_images = overlay_masks_on_images(X_test, predictions)

# Display some results
num_examples_to_show = 5
plt.figure(figsize=(15, 5))
for i in range(num_examples_to_show):
    plt.subplot(2, num_examples_to_show, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(2, num_examples_to_show, i + 1 + num_examples_to_show)
    plt.imshow(overlayed_images[i], cmap='gray')
    plt.title('Overlayed Mask')

plt.show()

# Optionally, you can calculate some metrics, like IoU or Dice coefficient
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calculate_dice(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    dice_score = (2. * np.sum(intersection)) / (np.sum(y_true) + np.sum(y_pred))
    return dice_score

# Example of calculating metrics (assuming you have ground truth masks)
# test_mask_dir = 'test_masks'  # Path to ground truth masks for the test images
# Y_test, _ = load_data(test_mask_dir)

# iou_scores = [calculate_iou(Y_test[i], predictions[i]) for i in range(len(Y_test))]
# dice_scores = [calculate_dice(Y_test[i], predictions[i]) for i in range(len(Y_test))]

# print('Mean IoU:', np.mean(iou_scores))
# print('Mean Dice:', np.mean(dice_scores))
