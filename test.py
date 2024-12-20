import matplotlib
matplotlib.use('TkAgg') 

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2 

model_path = 'unet_breast_segmentation.keras'
model = load_model(model_path)

def load_data(image_dir, img_size=(224, 224)):
    images = []
    image_names = []
    for img_name in os.listdir(image_dir):
        img = tf.keras.preprocessing.image.load_img(os.path.join(image_dir, img_name), target_size=img_size, color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0 
        images.append(img)
        image_names.append(img_name)
    
    return np.array(images), image_names

test_image_dir = 'normal' 

X_test, test_image_names = load_data(test_image_dir)

predictions = model.predict(X_test)

threshold = 0.5
predictions = (predictions > threshold).astype(np.uint8)

output_dir = 'predicted_masks'
os.makedirs(output_dir, exist_ok=True)
for i, pred in enumerate(predictions):
    pred = (pred * 255).astype(np.uint8) 
    pred_image = tf.keras.preprocessing.image.array_to_img(pred)
    pred_image.save(os.path.join(output_dir, f'pred_{test_image_names[i]}'))

def save_segmented_parts(images, masks, original_image_names, save_dir='segmented_parts'):
    os.makedirs(save_dir, exist_ok=True)
    for img, mask, img_name in zip(images, masks, original_image_names):
        img = (img * 255).astype(np.uint8) 
        img = img.squeeze() 
        mask = mask.squeeze() 
        segmented_part = cv2.bitwise_and(img, img, mask=mask)
        segmented_part = np.expand_dims(segmented_part, axis=-1)
        segmented_img = tf.keras.preprocessing.image.array_to_img(segmented_part)
        segmented_img.save(os.path.join(save_dir, f'seg_{img_name}'))

save_segmented_parts(X_test, predictions, test_image_names)

def overlay_masks_on_images(images, masks, alpha=0.5):
    overlayed_images = []
    for img, mask in zip(images, masks):
        img = img.squeeze() 
        mask = mask.squeeze()
        overlay = np.where(mask == 1, img * 0.5 + 0.5, img)
        overlayed_images.append(overlay)
    return overlayed_images

overlayed_images = overlay_masks_on_images(X_test, predictions)

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
