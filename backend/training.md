### Model Training done using Google Collab 

# ✅ STEP 1: Install required libraries
!pip install tensorflow opencv-python matplotlib

# ✅ STEP 2: Upload the dataset (ZIP file format)
from google.colab import files
uploaded = files.upload()  # Upload DevanagariHandwrittenCharacterDataset.zip

# ✅ STEP 3: Unzip the dataset
import zipfile
import os

zip_path = next(iter(uploaded))
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(".")

dataset_dir = "DevanagariHandwrittenCharacterDataset"
print("Extracted folders:", os.listdir(dataset_dir))

# ✅ STEP 4: Load & preprocess the data
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(32, 32)):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    label_dict = {name: idx for idx, name in enumerate(class_names)}

    for label_name in class_names:
        class_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(class_dir): continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label_dict[label_name])

    images = np.array(images, dtype='float32') / 255.0
    images = np.expand_dims(images, axis=-1)
    labels = to_categorical(labels, num_classes=len(class_names))
    return images, labels, class_names

train_dir = os.path.join(dataset_dir, "Train")
test_dir = os.path.join(dataset_dir, "Test")

X_train, y_train, class_names = load_data(train_dir)
X_test, y_test, _ = load_data(test_dir)

print(f"Training samples: {X_train.shape}, Classes: {len(class_names)}")

# ✅ STEP 5: Define a CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ✅ STEP 6: Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# ✅ STEP 7: Save and download the model
model.save("devanagari_character_model.h5")

# Optional: download model to local system
from google.colab import files
files.download("devanagari_character_model.h5")



#### Testing Recognition

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from google.colab import files
import os

# ✅ STEP 1: Recreate the class_names from folder
dataset_dir = "DevanagariHandwrittenCharacterDataset"
train_dir = os.path.join(dataset_dir, "Train")
class_names = sorted(os.listdir(train_dir))

# ✅ STEP 2: Load the trained model
model = load_model("devanagari_character_model.h5")

# ✅ STEP 3: Upload a test image
uploaded = files.upload()

for file_name in uploaded:
    print(f"Classifying: {file_name}")

    # Load and preprocess image
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension

    # Predict
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]

    print(f"Predicted character: {predicted_label}")

    # Visualize
    plt.imshow(cv2.imread(file_name), cmap='gray')
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')
    plt.show()
