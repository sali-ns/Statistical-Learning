# Load necessary libraries
library(data.table)  # For fast data manipulation
library(ggplot2)     # For visualization

# Load the dataset using fread for faster loading
data <- fread("creditcard.csv")

# Step 1: Data cleaning
data_cleaned <- na.omit(data)

# Identify and remove constant columns
constant_cols <- sapply(data_cleaned[, -ncol(data_cleaned), with = FALSE], function(x) var(x, na.rm = TRUE) == 0)
if (any(constant_cols)) {
  cat("Removing constant columns:", names(data_cleaned)[constant_cols], "\n")
  data_cleaned <- data_cleaned[, !constant_cols, with = FALSE]
}

# Step 2: Scale the features (excluding the target 'Class')
scaled_data <- scale(data_cleaned[, -ncol(data_cleaned), with = FALSE])  # Exclude the target 'Class'

# Step 3: Perform PCA
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)

# Step 4: Variance Explained by Each Principal Component
explained_variance <- summary(pca_result)$importance[2, ]  # Proportion of variance explained
cumulative_variance <- cumsum(explained_variance)  # Cumulative variance

# Print explained and cumulative variance
cat("Explained variance by each component:\n")
print(explained_variance)

cat("\nCumulative explained variance:\n")
print(cumulative_variance)

# Step 5: Prepare PCA results for visualization
pca_data <- as.data.table(pca_result$x[, 1:2])  # First two principal components
pca_data[, Class := data_cleaned$Class]  # Add the Class column for coloring in the plot

# Visualize PCA results
ggplot(pca_data, aes(x = PC1, y = PC2, color = Class)) +
  geom_point(alpha = 0.5) +
  labs(title = "PCA of Credit Card Transactions") +
  theme_minimal()

# Step 6: Visualize Explained Variance
variance_plot <- ggplot(data.frame(PC = seq_along(explained_variance), Variance = explained_variance),
                        aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Variance Explained by Each Principal Component", x = "Principal Component", y = "Proportion of Variance Explained") +
  theme_minimal()
library(grid)
grid.newpage()
print(variance_plot)

# Step 7: Visualize Cumulative Explained Variance
cumulative_variance_plot <- ggplot(data.frame(PC = seq_along(cumulative_variance), Cumulative_Variance = cumulative_variance),
                                   aes(x = PC, y = Cumulative_Variance)) +
  geom_line(color = "red", size = 1) +
  labs(title = "Cumulative Variance Explained by Principal Components", x = "Principal Component", y = "Cumulative Variance Explained") +
  theme_minimal()

print(cumulative_variance_plot)
# Step 1: Visualize the first few principal components (e.g., PC1, PC2, PC3)
# Prepare PCA results for visualization with more components
pca_data_extended <- as.data.table(pca_result$x[, 1:3])  # Taking first three components
pca_data_extended[, Class := data_cleaned$Class]  # Add the Class column for coloring

# Visualize first three principal components
library(plotly)
pca_plot <- plot_ly(pca_data_extended, x = ~PC1, y = ~PC2, z = ~PC3, color = ~Class,
                    colors = c("#636EFA", "#EF553B"),
                    type = "scatter3d", mode = "markers",
                    marker = list(size = 3, opacity = 0.7)) %>%
  layout(title = "3D PCA of Credit Card Transactions",
         scene = list(xaxis = list(title = 'PC1'),
                      yaxis = list(title = 'PC2'),
                      zaxis = list(title = 'PC3')))
pca_plot
# Step 2: Elbow method to determine optimal number of components
elbow_plot <- ggplot(data.frame(PC = seq_along(cumulative_variance),
                                Cumulative_Variance = cumulative_variance),
                     aes(x = PC, y = Cumulative_Variance)) +
  geom_line() +
  geom_point() +
  labs(title = "Elbow Method for Principal Components",
       x = "Number of Principal Components",
       y = "Cumulative Variance Explained") +
  theme_minimal()

print(elbow_plot)

# Step 3: Calculate feature importance
loading_scores <- pca_result$rotation[, 1:2]  # Loadings for PC1 and PC2
importance <- as.data.table(loading_scores)
importance[, Feature := rownames(importance)]
importance_long <- melt(importance, id.vars = "Feature", variable.name = "PC", value.name = "Importance")

# Visualize feature importance for the first two principal components
importance_plot <- ggplot(importance_long, aes(x = reorder(Feature, Importance), y = Importance, fill = PC)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(title = "Feature Importance for Principal Components", x = "Features", y = "Importance") +
  theme_minimal()

print(importance_plot)
# Logistic Regression on PCA components

# Split the dataset into training and testing sets
set.seed(123)
train_index <- createDataPartition(data_cleaned$Class, p = 0.7, list = FALSE)
train_data <- pca_data[, 1:2][train_index, ]
train_labels <- data_cleaned$Class[train_index]
test_data <- pca_data[, 1:2][-train_index, ]
test_labels <- data_cleaned$Class[-train_index]

# Train a logistic regression model
model <- glm(train_labels ~ PC1 + PC2, data = train_data, family = "binomial")

# Make predictions on the test set
predictions <- predict(model, newdata = test_data, type = "response")
predicted_classes <- ifelse(predictions > 0.5, "1", "0")

# Evaluate model performance
confusionMatrix(as.factor(predicted_classes), as.factor(test_labels))


