#!/usr/bin/env python
# coding: utf-8

# In[1]:


import deeplake
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Input,BatchNormalization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import cv2


# In[2]:


# Load the FER2013 dataset
ds = deeplake.load('hub://activeloop/fer2013-train')


# In[3]:


ds.info


# In[4]:


print(ds.summary())


# In[5]:


# Extract class names from the dataset metadata
metadata = ds.info['classes']
class_names = [name.split('=')[1] for name in metadata.split(', ')]
print(class_names)


# In[6]:


#Preprocess the dataset
def generator():
    for sample in ds:
        image = tf.reshape(sample['images'], (48, 48, 1))  # Reshape the image
        label = sample['labels'][0]  # Extract the label value
        yield image, label

output_signature = (
    tf.TensorSpec(shape=(48, 48, 1), dtype=tf.uint8),  # Adjust the shape and dtype based on your data
    tf.TensorSpec(shape=(), dtype=tf.int64)  # Adjust the shape and dtype based on your data
)

tf_ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

# Filter the dataset for "Sad" and "Happy" classes
def filter_classes(image, label):
    return tf.logical_or(tf.equal(label, 3), tf.equal(label, 4))  # 3 corresponds to "Happy", 4 corresponds to "Sad"

filtered_ds = tf_ds.filter(filter_classes)
batch_size = 32
# Batch and prefetch the dataset
filtered_ds = filtered_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# In[7]:


# Visualize some images from the dataset
def visualize_images(filtered_ds, class_names, num_images=9):

    plt.figure(figsize=(10, 10))
    
    for images, labels in filtered_ds.take(1):  # Take 1 batch from the dataset
        for i in range(min(num_images, len(images))):  # Ensure num_images doesn't exceed batch size
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8").squeeze(), cmap='gray')
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")

    plt.show()

# Visualize 9 images from the filtered dataset
visualize_images(filtered_ds, class_names, num_images=9)


# In[8]:


# Assuming labels for happy and sad are 3 and 4 respectively
happy_label = 3
sad_label = 4


# In[9]:


# Extract images and labels for happy and sad
images = []
labels = []

for sample in ds:
    label = sample['labels'].numpy()
    if label == happy_label or label == sad_label:
        images.append(sample['images'].numpy())
        labels.append(1 if label == happy_label else 0)  # 1 for happy, 0 for sad

images = np.array(images)
labels = np.array(labels)


# In[10]:


# Normalize the images
images = images / 255.0


# In[11]:


# Reshape images if necessar
images = images.reshape(-1, 48, 48, 1)  # Assuming the images are 48x48 pixels


# In[12]:


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")


# In[13]:


# Define the model architecture
model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(8, (3, 3), padding="same", activation='relu'),
    BatchNormalization(),  # Add batch normalization layer
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.1),  # Dropout layer to prevent overfitting
    Conv2D(16, (3, 3), padding="same", activation='relu'),
    BatchNormalization(),  # Add batch normalization layer
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.2),  # Dropout layer to prevent overfitting
    Conv2D(32, (3, 3), padding="same", activation='relu'),
    BatchNormalization(),  # Add batch normalization layer
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), padding="same", activation='relu'),
    BatchNormalization(),  # Add batch normalization layer
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout layer to prevent overfitting
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history =model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))


# In[14]:


# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[15]:


fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=10)
plt.legend(loc="upper left")
plt.show()


# In[16]:


# Define the class names
classes = ['Sad', 'Happy']

# Make predictions on the validation set
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()  # Convert probabilities to binary labels

# Calculate the confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[17]:


# Print classification report
print(classification_report(y_val, y_pred, target_names=classes, zero_division=1))


# In[22]:


# Load the test dataset
ds_test = deeplake.load('hub://activeloop/fer2013-public-test')

# Filter the dataset for "Happy" and "Sad" classes
happy_label = 3  # Assuming label 3 is "Happy"
sad_label = 4    # Assuming label 4 is "Sad"

def filter_classes(sample):
    label = sample['labels'].numpy()
    return label == happy_label or label == sad_label

# Extract images and labels for happy and sad
images = []
labels = []

for sample in ds_test:
    label = sample['labels'].numpy()
    if filter_classes(sample):
        images.append(sample['images'].numpy())
        labels.append(1 if label == happy_label else 0)  # 1 for happy, 0 for sad

images = np.array(images)
labels = np.array(labels)

# Normalize the images
images = images / 255.0

# Reshape images if necessary (e.g., add a channel dimension for grayscale images)
images = images.reshape(-1, 48, 48, 1)  # Assuming the images are 48x48 pixels

# Define preprocess function
def preprocess_image(image):
    return image / 255.0

# Preprocess and predict
for i, image in enumerate(images[:10]):  # Displaying the first 10 images for brevity
    input_image = preprocess_image(image)

    # Display the original image
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

    # Display the preprocessed image
    plt.imshow(input_image.squeeze(), cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')
    plt.show()

    # Make a prediction using the model
    yhat = model.predict(np.expand_dims(input_image, axis=0))

    # Interpret the prediction
    predicted_class = 'Sad' if yhat[0] > 0.5 else 'Happy'
    actual_class = 'Happy' if labels[i] == 1 else 'Sad'
    print(f'Predicted class is {predicted_class}, Actual class is {actual_class}')

