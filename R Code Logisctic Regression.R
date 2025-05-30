# ========================================
# Logistic Regression Analysis on Diabetes
# ========================================

# --- 0. Load Required Libraries ---
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(ResourceSelection)
library(pscl)
library(car)

# --- 1. Data Preprocessing ---

# Load data
data <- read_csv("D:/S2 STATISTIKA ZHERYL/Praktisi Mengajar/Pertemuan 2/archive(4)/data_diabetes_BRFSS2015.csv")

# Select relevant columns
data_selected <- data %>%
  select(Diabetes_binary, HighBP, HighChol, PhysActivity, Age, Fruits, Veggies)

# Remove missing values
data_clean <- na.omit(data_selected)

# --- 2. Exploratory Data Analysis (EDA) ---

# Summary
summary(data_clean)

# Helper function for proportional labels
get_prop_labels <- function(df, xvar, fillvar) {
  df %>%
    group_by(across(all_of(c(xvar, fillvar)))) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(across(all_of(xvar))) %>%
    mutate(prop = n / sum(n),
           label = scales::percent(prop, accuracy = 1))
}

# List of categorical variables for stacked bar plots
cat_vars <- c("HighBP", "HighChol", "PhysActivity", "Fruits", "Veggies")

# Plot each categorical variable
for (var in cat_vars) {
  labels_df <- get_prop_labels(data_clean, "Diabetes_binary", var)
  ggplot(data_clean, aes(x = factor(Diabetes_binary, labels = c("No", "Yes")), fill = .data[[var]])) +
    geom_bar(position = "fill") +
    geom_text(data = labels_df,
              aes(x = factor(Diabetes_binary, labels = c("No", "Yes")), y = prop, label = label, group = .data[[var]]),
              position = position_fill(vjust = 0.5), inherit.aes = FALSE, size = 3) +
    labs(title = paste(var, "by Diabetes Status"),
         x = "Diabetes", y = "Proportion", fill = var)
}

# Boxplot: Age vs Diabetes
ggplot(data_clean, aes(x = factor(Diabetes_binary, labels = c("No", "Yes")),
                       y = Age, fill = factor(Diabetes_binary))) +
  geom_boxplot() +
  labs(title = "Age Distribution by Diabetes Status",
       x = "Diabetes", y = "Age") +
  theme(legend.position = "none")

# --- 3. Data Splitting (80:20) ---

set.seed(123)
split_index <- createDataPartition(data_clean$Diabetes_binary, p = 0.8, list = FALSE)
train_data <- data_clean[split_index, ]
test_data  <- data_clean[-split_index, ]

# --- 4. Logistic Regression Model (Training Data) ---

# Build model
model <- glm(Diabetes_binary ~ HighBP + HighChol + PhysActivity + Age + Fruits + Veggies,
             data = train_data, family = binomial)

# Model Summary
summary(model)

# Odds Ratios
exp(coef(model))

# Simultaneous Test (Likelihood Ratio Test)
anova(model, test = "Chisq")

# Goodness-of-Fit: Hosmer-Lemeshow
hoslem.test(as.numeric(train_data$Diabetes_binary), fitted(model))

# Pseudo R²
pR2(model)

# Multicollinearity Check: VIF
vif(model)

# --- 5. Model Evaluation on Test Set ---

# Predict
pred_probs  <- predict(model, newdata = test_data, type = "response")
pred_class  <- ifelse(pred_probs > 0.5, 1, 0)

# Ensure levels match
true <- factor(test_data$Diabetes_binary, levels = c(0, 1))
pred <- factor(pred_class, levels = c(0, 1))

# Confusion Matrix
conf_matrix <- confusionMatrix(pred, true, positive = "1")

# Extract Evaluation Metrics
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
accuracy    <- conf_matrix$overall["Accuracy"]

# Print Results
cat("Sensitivity :", round(sensitivity, 3), "\n")
cat("Specificity :", round(specificity, 3), "\n")
cat("Accuracy    :", round(accuracy, 3), "\n")
