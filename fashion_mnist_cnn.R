# Load required libraries
library(keras)
library(ggplot2)

# Create a fashion MNIST classifier class
FashionMNISTModel <- function() {
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', 
                  input_shape = c(28, 28, 1)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')
  
  model %>% compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = c('accuracy')
  )
  
  class_names <- c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
  
  list(
    model = model,
    class_names = class_names,
    history = NULL,
    
    train = function(self, train_images, train_labels, epochs = 10, validation_data = NULL) {
      self$history <- self$model %>% fit(
        train_images, train_labels,
        epochs = epochs,
        validation_data = validation_data,
        verbose = 1
      )
      return(self$history)
    },
    
    evaluate_model = function(self, test_images, test_labels) {
      score <- self$model %>% evaluate(test_images, test_labels, verbose = 0)
      return(score)
    },
    
    predict_images = function(self, images) {
      predictions <- self$model %>% predict(images)
      return(predictions)
    },
    
    get_predictions_details = function(self, predictions, true_labels) {
      predicted_classes <- apply(predictions, 1, which.max) - 1
      results <- list()
      
      for (i in 1:nrow(predictions)) {
        result <- list(
          true_label = true_labels[i],
          true_class_name = self$class_names[true_labels[i] + 1],
          predicted_class = predicted_classes[i],
          predicted_class_name = self$class_names[predicted_classes[i] + 1],
          confidence = max(predictions[i,]) * 100,
          all_probabilities = predictions[i,]
        )
        results[[i]] <- result
      }
      
      return(results)
    }
  )
}

# Start logging
start_time <- Sys.time()
log_file <- paste0("r_output_", format(start_time, "%Y%m%d_%H%M%S"), ".txt")
sink(log_file, split = TRUE)  # split=TRUE sends output to both console and file

cat("============================================================\n")
cat("FASHION MNIST CLASSIFICATION - R IMPLEMENTATION\n")
cat("============================================================\n")
cat("Log file:", log_file, "\n")
cat("Start time:", format(start_time, "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n")

# Load Fashion MNIST dataset
cat("Loading Fashion MNIST dataset...\n")
fashion_mnist <- dataset_fashion_mnist()
train_images <- fashion_mnist$train$x
train_labels <- fashion_mnist$train$y
test_images <- fashion_mnist$test$x
test_labels <- fashion_mnist$test$y

cat("Training images dimensions:", dim(train_images), "\n")
cat("Training labels length:", length(train_labels), "\n")
cat("Test images dimensions:", dim(test_images), "\n")
cat("Test labels length:", length(test_labels), "\n")

# Define class names
class_names <- c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

cat("\nClass names:\n")
for (i in 1:length(class_names)) {
  cat(sprintf("  %d: %s\n", i-1, class_names[i]))
}

# Preprocess the data
cat("\nPreprocessing data...\n")
# Normalize pixel values
train_images <- train_images / 255
test_images <- test_images / 255

# Reshape data to add channel dimension
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

cat("After reshaping - Training images dimensions:", dim(train_images), "\n")
cat("After reshaping - Test images dimensions:", dim(test_images), "\n")

# Build the CNN model with 6 layers using class
cat("\nBuilding the CNN model with 6 layers using FashionMNISTModel class...\n")
fashion_model <- FashionMNISTModel()

cat("\nModel architecture:\n")
summary(fashion_model$model)

# Compile the model
fashion_model$model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

cat("\nModel compiled with Adam optimizer and sparse categorical crossentropy loss\n")

# Train the model
cat("\nTraining the model for 10 epochs...\n")
history <- fashion_model$train(fashion_model, train_images, train_labels, epochs = 10,
                              validation_data = list(test_images, test_labels))

cat("\nTraining completed!\n")

# Evaluate the model
cat("\nEvaluating the model on test data...\n")
score <- fashion_model$evaluate_model(fashion_model, test_images, test_labels)
cat(sprintf('Test loss: %.4f\n', score$loss))
cat(sprintf('Test accuracy: %.4f\n', score$acc))

# Make predictions on two images
cat("\nMaking predictions on two sample images...\n")
predictions <- fashion_model$predict_images(fashion_model, test_images[1:2,,,])
prediction_details <- fashion_model$get_predictions_details(fashion_model, predictions, test_labels[1:2])

cat("\nFirst image prediction probabilities:\n")
for (i in 1:length(class_names)) {
  cat(sprintf("  %s: %.2f%%\n", class_names[i], predictions[1,i] * 100))
}

cat("\nSecond image prediction probabilities:\n")
for (i in 1:length(class_names)) {
  cat(sprintf("  %s: %.2f%%\n", class_names[i], predictions[2,i] * 100))
}

# Display the results
png('predictions.png', width = 800, height = 400)
par(mfrow = c(1, 2))
for (i in 1:2) {
  # Plot image
  image_array <- test_images[i,,,1]
  image(1:28, 1:28, image_array, col = gray.colors(256), 
        main = paste("True:", fashion_model$class_names[test_labels[i] + 1], 
                     "\nPredicted:", fashion_model$class_names[which.max(predictions[i,])]),
        xlab = "", ylab = "", axes = FALSE)
  box()
}
par(mfrow = c(1, 1))
dev.off()
cat("\nPrediction visualization saved as predictions.png\n")

# Print the predictions
cat("\nDETAILED PREDICTION RESULTS:\n")
cat("==================================================\n")
for (i in 1:2) {
  detail <- prediction_details[[i]]
  cat(sprintf("\nImage %d:\n", i))
  cat(sprintf("True label: %s (%d)\n", detail$true_class_name, detail$true_label))
  cat(sprintf("Predicted: %s (%d)\n", detail$predicted_class_name, detail$predicted_class))
  cat(sprintf("Confidence: %.2f%%\n", detail$confidence))
  
  # Show top 3 probabilities
  sorted_probs <- sort(detail$all_probabilities, decreasing = TRUE, index.return = TRUE)
  cat("Top 3 predictions:\n")
  for (j in 1:3) {
    idx <- sorted_probs$ix[j]
    cat(sprintf("  %s: %.2f%%\n", fashion_model$class_names[idx], detail$all_probabilities[idx] * 100))
  }
}

end_time <- Sys.time()
cat("\n============================================================\n")
cat("PREDICTION COMPLETE!\n")
cat("End time:", format(end_time, "%Y-%m-%d %H:%M:%S"), "\n")
cat("Total execution time:", format(end_time - start_time), "\n")
cat("============================================================\n")

# Stop logging
sink()

cat("All outputs have been saved to", log_file, "\n")
cat("Prediction visualization saved as predictions.png\n")