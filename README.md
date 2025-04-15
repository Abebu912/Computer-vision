# Computer-vision
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the augmentation folder
augmentation_folder = 'C:/Users/user/Desktop/facial expression recognizer project/Training'
# Create the ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_images():
    # Loop through each folder (student) in the augmentation folder
    for student_folder in os.listdir(augmentation_folder):
        student_path = os.path.join(augmentation_folder, student_folder)
        
        if os.path.isdir(student_path):
            # Create a directory for augmented images if it doesn't exist
            augmented_dir = os.path.join(student_path, 'augmented')
            if not os.path.exists(augmented_dir):
                os.makedirs(augmented_dir)
            
            # Loop through each image in the student folder
            for filename in os.listdir(student_path):
                img_path = os.path.join(student_path, filename)
                
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = np.expand_dims(img, axis=0)

                    # Generate augmented images and save them
                    i = 0
                    for batch in datagen.flow(img, batch_size=1, save_to_dir=augmented_dir, save_prefix='aug', save_format='jpg'):
                        i += 1
                        if i > 20:  # Save 20 augmented images per original image
                            break

# Call the augmentation function
augment_images()
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Define data generators with data augmentation for the training set
train_dir = 'C:/Users/user/Desktop/facial expression recognizer project/Training'
val_dir = 'C:/Users/user/Desktop/facial expression recognizer project/Testing'

# Apply data augmentation to the training set to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # Random rotations between 0-20 degrees
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2, # Random vertical shifts
    shear_range=0.2,        # Random shearing
    zoom_range=0.2,         # Random zoom
    horizontal_flip=True,   # Random horizontal flips
    fill_mode='nearest'     # Fill any new pixels after transformations
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for validation

# Create the generators to load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224 pixels
    color_mode='grayscale',  # Use grayscale images
    class_mode='categorical',  # For multi-class classification
    batch_size=32,           # Use a reasonable batch size for smaller datasets
    shuffle=True             # Shuffle the data to ensure randomness
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32
)

# Build the model using the Functional API
input_layer = layers.Input(shape=(224, 224, 1))  # Input layer with shape (224, 224, 1)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)  # First convolutional layer
x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Max pooling
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # Second convolutional layer
x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Max pooling
x = layers.Flatten()(x)  # Flatten the 2D features to 1D
x = layers.Dense(128, activation='relu')(x)  # Fully connected layer with 128 neurons
x = layers.Dropout(0.5)(x)  # Dropout layer to prevent overfitting
output_layer = layers.Dense(7, activation='softmax')(x)  # Output layer for 7 classes (facial expressions)

# Create the model
model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# EarlyStopping to stop training if the validation loss doesn't improve
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[early_stopping]
)# Train the model with the data generators and apply EarlyStopping
# Save the trained model
base_dir = 'C:/Users/user/Desktop/facial expression recognizer project'
model.save(os.path.join(base_dir, "student_expression_model.keras"))
print("Learning ends successfully!")
import os
import numpy as np
from tkinter import Tk, Label, Button, filedialog
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import ImageTk, Image
import matplotlib.pyplot as plt

# Path to the saved model
model_path = 'C:/Users/user/Desktop/facial expression recognizer project/student_expression_model.keras'
# Load the trained model
model = load_model(model_path)

# Class labels for the facial expressions
class_labels = ['abebe', 'abebu', 'firawol', 'habtamu', 'husein', 'jegnaw', 'meswat']

# Set a threshold for minimum confidence to accept a prediction
confidence_threshold = 0.6  # Adjust this threshold as needed (between 0 and 1)

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')  # Resize to 224x224
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for the batch size
    img_array = img_array / 255.0  # Rescale the image
    return img_array

# Function to predict the person from an image
def predict_person(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]  # Get the index of the max probability
    confidence = prediction[0][predicted_class_idx]  # Get the confidence of the predicted class
    predicted_class = class_labels[predicted_class_idx]
    
    # Check if the confidence is above the threshold
    if confidence < confidence_threshold:
        return "Unknown Person", confidence  # If the confidence is low, label as "Unknown Person"
    else:
        return predicted_class, confidence

# Function to open file dialog and select an image for prediction
def upload_image():
    file_path = filedialog.askopenfilename()  # Open file dialog to select an image
    if file_path:
        # Display image preview on the GUI
        img = Image.open(file_path)
        img = img.resize((150, 150))  # Resize image for better display in the GUI
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        #Predict the person
        predicted_person, confidence = predict_person(file_path)
        
        # Display the result
        if predicted_person == "Unknown Person":
            result_label.config(text=f"Prediction: {predicted_person} (Confidence: {confidence*100:.2f}%)")
        else:
            result_label.config(text=f"Predicted Person: {predicted_person} (Confidence: {confidence*100:.2f}%)")

# Set up the Tkinter window
window = Tk()
window.title("group 3 face test")
window.geometry("400x400")

# Create a label for showing the image
panel = Label(window)
panel.pack(padx=20, pady=20)

# Create a button to upload an image
upload_button = Button(window, text="Upload Image", width=20, command=upload_image)
upload_button.pack(pady=10)

# Create a label for displaying the prediction result
result_label = Label(window, text="Predicted Expression: ", font=("Arial", 14))
result_label.pack(pady=10)

# Start the Tkinter GUI loop
window.mainloop()
