# 1. Load necessary libraries
library(caret)          # For model training and evaluation
library(randomForest)    # For Random Forest classifier
library(dplyr)           # For data manipulation
library(ggplot2)         # For visualization

# 2. Load the dataset
# Note: Change the path if needed
data <- read.csv("creditcard.csv")

# 3. Data exploration
str(data)          # Check structure of the dataset
summary(data)      # Get basic statistics
table(data$Class)  # Check the distribution of the target variable (Class)

# Check for missing values
missing_percentage <- (colSums(is.na(data)) / nrow(data)) * 100
print(missing_percentage)

# Visualize the class distribution
ggplot(data, aes(x = as.factor(Class))) +
  geom_bar() +
  ggtitle("Class Distribution") +
  xlab("Class") + ylab("Count")

# 4. Split the data into train and test sets
set.seed(123)  # Ensure reproducibility
index <- createDataPartition(data$Class, p = 0.7, list = FALSE)

train <- data[index,]
test <- data[-index,]

# Check the distribution of the target variable in train and test sets
table(train$Class)
table(test$Class)

# 5. Preprocess Data
# Normalize/scale the features (excluding the target 'Class')
train_scaled <- train
test_scaled <- test

train_scaled[, 1:30] <- scale(train[, 1:30])
test_scaled[, 1:30] <- scale(test[, 1:30])

# 6. Build a Random Forest Model
set.seed(123)
model_rf <- randomForest(Class ~ ., data = train_scaled, ntree = 100, importance = TRUE)

# Print model details
print(model_rf)

# 7. Evaluate the model on the test set
predictions_rf <- predict(model_rf, newdata = test_scaled)
confusionMatrix(predictions_rf, test_scaled$Class)

# 8. Feature importance visualization
varImpPlot(model_rf)

# 9. Calculate performance metrics
# Additional metrics can be calculated if needed
pred_probs <- predict(model_rf, newdata = test_scaled, type = "prob")
roc_auc <- roc.curve(scores.class0 = pred_probs[,1],
                     scores.class1 = pred_probs[,2],
                     curve = TRUE)
plot(roc_auc, main = "ROC Curve")
