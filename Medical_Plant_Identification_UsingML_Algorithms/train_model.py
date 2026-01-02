import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define constants
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 50
DATA_DIR = 'dataset'  # Directory containing plant images organized by class

def create_model(num_classes):
    """Create CNN model for plant identification"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def prepare_data():
    """Prepare data generators for training and validation"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig('training_history.png')
    plt.show()

def main():
    """Main training function"""
    # Check if dataset directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Dataset directory '{DATA_DIR}' not found.")
        print("Please organize your plant images in folders named by plant type.")
        print("Example structure:")
        print("dataset/")
        print("  ├── AMLA/")
        print("  ├── BASIL/")
        print("  └── ...")
        return

    # Prepare data
    try:
        train_generator, validation_generator = prepare_data()
        num_classes = len(train_generator.class_indices)
        print(f"Found {num_classes} plant classes: {list(train_generator.class_indices.keys())}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        return

    # Create and train model
    model = create_model(num_classes)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # Save the model
    model.save('plant_identification_model.h5')

    # Save class indices for prediction
    np.save('class_indices.npy', train_generator.class_indices)

    # Plot training history
    plot_training_history(history)

    print("Model training completed!")
    print("Model saved as 'plant_identification_model.h5'")
    print("Class indices saved as 'class_indices.npy'")

if __name__ == "__main__":
    main()
