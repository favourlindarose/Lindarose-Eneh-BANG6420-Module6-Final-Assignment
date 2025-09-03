# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# Create a function to print to both console and file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# Custom class for Fashion MNIST CNN
class FashionMNISTClassifier:
    def __init__(self):
        self.model = None
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.build_model()
        self.history = None
    
    def build_model(self):
        """Build the 6-layer CNN model"""
        self.model = models.Sequential()
        # Layer 1: Convolutional layer
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        # Layer 2: Pooling layer
        self.model.add(layers.MaxPooling2D((2, 2)))
        # Layer 3: Convolutional layer
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # Layer 4: Pooling layer
        self.model.add(layers.MaxPooling2D((2, 2)))
        # Layer 5: Convolutional layer
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # Layer 6: Dense output layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))
        
        self.model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
    
    def train(self, train_images, train_labels, epochs=10, validation_data=None):
        """Train the model"""
        self.history = self.model.fit(train_images, train_labels, epochs=epochs,
                                    validation_data=validation_data)
        return self.history
    
    def evaluate_model(self, test_images, test_labels):
        """Evaluate the model on test data"""
        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=0)
        return test_loss, test_acc
    
    def predict_images(self, images):
        """Make predictions on images"""
        return self.model.predict(images)
    
    def get_predictions_details(self, predictions, true_labels):
        """Get detailed prediction information"""
        predicted_classes = np.argmax(predictions, axis=1)
        results = []
        
        for i in range(len(predictions)):
            result = {
                'true_label': true_labels[i],
                'true_class': self.class_names[true_labels[i]],
                'predicted_class': predicted_classes[i],
                'predicted_class_name': self.class_names[predicted_classes[i]],
                'confidence': np.max(predictions[i]) * 100,
                'all_probabilities': predictions[i]
            }
            results.append(result)
        
        return results

# Redirect output to file
log_file = f"python_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
sys.stdout = Logger(log_file)

print("=" * 60)
print("FASHION MNIST CLASSIFICATION - PYTHON IMPLEMENTATION")
print("=" * 60)
print(f"Log file: {log_file}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Load the Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Define class names for the 10 categories
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("\nClass names:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# Preprocess the data
print("\nPreprocessing data...")
# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape data to add a channel dimension (required for CNN)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

print(f"After reshaping - Training images shape: {train_images.shape}")
print(f"After reshaping - Test images shape: {test_images.shape}")

# Build the CNN model with 6 layers using custom class
print("\nBuilding the CNN model with 6 layers using FashionMNISTClassifier class...")
classifier = FashionMNISTClassifier()

print("\nModel architecture:")
classifier.model.summary()

# Compile the model
classifier.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nModel compiled with Adam optimizer and sparse categorical crossentropy loss")

# Train the model
print("\nTraining the model for 10 epochs...")
history = classifier.train(train_images, train_labels, epochs=10, 
                          validation_data=(test_images, test_labels))

print("\nTraining completed!")

# Evaluate the model
print("\nEvaluating the model on test data...")
test_loss, test_acc = classifier.evaluate_model(test_images, test_labels)
print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.4f}')

# Make predictions on two images
print("\nMaking predictions on two sample images...")
predictions = classifier.predict_images(test_images[:2])
prediction_details = classifier.get_predictions_details(predictions, test_labels[:2])

print(f"\nFirst image prediction probabilities:")
for i, prob in enumerate(predictions[0]):
    print(f"  {class_names[i]}: {prob * 100:.2f}%")

print(f"\nSecond image prediction probabilities:")
for i, prob in enumerate(predictions[1]):
    print(f"  {class_names[i]}: {prob * 100:.2f}%")

# Display the results
plt.figure(figsize=(10, 5))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"True: {class_names[test_labels[i]]}\nPredicted: {class_names[np.argmax(predictions[i])]}")
    plt.axis('off')
    
prediction_image = 'predictions.png'
plt.savefig(prediction_image)
print(f"\nPrediction visualization saved as {prediction_image}")

# Print the predictions
print("\nDETAILED PREDICTION RESULTS:")
print("=" * 50)
for i, detail in enumerate(prediction_details):
    print(f"\nImage {i+1}:")
    print(f"True label: {detail['true_class']} ({detail['true_label']})")
    print(f"Predicted: {detail['predicted_class_name']} ({detail['predicted_class']})")
    print(f"Confidence: {detail['confidence']:.2f}%")
    
    # Show top 3 probabilities
    top_3_indices = np.argsort(detail['all_probabilities'])[-3:][::-1]
    print("Top 3 predictions:")
    for j in top_3_indices:
        print(f"  {class_names[j]}: {detail['all_probabilities'][j] * 100:.2f}%")

print("\n" + "=" * 60)
print("PREDICTION COMPLETE!")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Close the log file
sys.stdout.close()
sys.stdout = sys.stdout.terminal

print(f"All outputs have been saved to {log_file}")
print(f"Prediction visualization saved as {prediction_image}")