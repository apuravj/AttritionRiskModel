# Load necessary library
library(readr)

# Load dataset
df <- read_csv(file.choose())  # Update with your local path
head(df)



################################################################################


# Convert Attrition and other categorical variables to factors
df$Attrition <- as.factor(df$Attrition)
df$BusinessTravel <- as.factor(df$BusinessTravel)
df$Department <- as.factor(df$Department)
df$Gender <- as.factor(df$Gender)
df$JobRole <- as.factor(df$JobRole)
df$MaritalStatus <- as.factor(df$MaritalStatus)
df$OverTime <- as.factor(df$OverTime)


################################################################################


df <- df %>%
  select(-c(EmployeeCount, EmployeeNumber, Over18, StandardHours))


################################################################################


# Check for missing values
colSums(is.na(df))


################################################################################


# Normalize numerical columns
numerical_cols <- sapply(df, is.numeric)
df[numerical_cols] <- scale(df[numerical_cols])


################################################################################


library(ggplot2)

# Attrition Distribution
ggplot(df, aes(x = Attrition)) +
  geom_bar(fill = "skyblue") +
  ggtitle("Attrition Distribution")


################################################################################


# Boxplot: Monthly Income vs Attrition
ggplot(df, aes(x = Attrition, y = MonthlyIncome)) +
  geom_boxplot(fill = "lightgreen") +
  ggtitle("Monthly Income vs Attrition")

# Bar plot: Overtime vs Attrition
ggplot(df, aes(x = OverTime, fill = Attrition)) +
  geom_bar(position = "dodge") +
  ggtitle("OverTime vs Attrition")


################################################################################


library(corrplot)

# Correlation matrix for numerical columns
corr_matrix <- cor(df[numerical_cols])
corrplot(corr_matrix, method = "circle")


################################################################################


# Load caret library
library(caret)

# Split dataset into training (70%) and test (30%)
set.seed(42)
trainIndex <- createDataPartition(df$Attrition, p = 0.7, list = FALSE)
train_data <- df[trainIndex, ]
test_data <- df[-trainIndex, ]


################################################################################


# Logistic Regression
log_model <- glm(Attrition ~ ., data = train_data, family = binomial)
summary(log_model)

# Predict and evaluate
pred_log <- predict(log_model, newdata = test_data, type = "response")
pred_class <- ifelse(pred_log > 0.5, "Yes", "No")

# Confusion Matrix
confusionMatrix(as.factor(pred_class), test_data$Attrition)


################################################################################


# Load randomForest library
library(randomForest)

# Train Random Forest
rf_model <- randomForest(Attrition ~ ., data = train_data, ntree = 100)
print(rf_model)

# Predict and evaluate
pred_rf <- predict(rf_model, newdata = test_data)
confusionMatrix(pred_rf, test_data$Attrition)


################################################################################


# AUC for Logistic Regression
library(pROC)

roc_curve <- roc(test_data$Attrition, as.numeric(pred_log))
auc(roc_curve)
plot(roc_curve)


################################################################################


# Export predictions and feature importance
predictions <- data.frame(Actual = test_data$Attrition, Predicted = pred_rf)
write.csv(predictions, "predictions.csv")

# Feature importance (for random forest)
importance_rf <- data.frame(Feature = rownames(importance(rf_model)), Importance = importance(rf_model)[, 1])
write.csv(importance_rf, "feature_importance.csv")