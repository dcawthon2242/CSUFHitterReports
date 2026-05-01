#===============================================================================
# Libraries
#===============================================================================
#download necessary libraries
install.packages("tidyverse")
install.packages("lightgbm")
install.packages("xgboost")
install.packages("rBayesianOptimization")
install.packages("caret")
install.packages("pROC")
install.packages("caTools")

# Load necessary libraries
library(tidyverse)
library(caTools)
library(lightgbm)
library(xgboost)
library(pROC)
library(rBayesianOptimization)
library(xgboost)
library(caret)

#===============================================================================
# Data Prep
#===============================================================================


#===============================================================================
# STANDARDIZE PITCH TYPES IN TRAINING DATA
#===============================================================================
D1TM25 <- D1TM25 %>%
  mutate(TaggedPitchType = case_when(
    TaggedPitchType %in% c("Changeup", "ChangeUp") ~ "ChangeUp",
    TaggedPitchType %in% c("Fastball", "FourSeamFastBall") ~ "Fastball",
    TaggedPitchType %in% c("Sinker", "TwoSeamFastBall") ~ "Sinker",
    TaggedPitchType == "OneSeamFastBall" ~ "Fastball",  # Merge rare types
    TaggedPitchType %in% c("Undefined", "Other") ~ "Other",
    TRUE ~ as.character(TaggedPitchType)
  ))

cat("Pitch types after standardization:\n")
print(table(D1TM25$TaggedPitchType))

D1TM25 <- D1TM25 %>%
  arrange(GameID, Inning, Top.Bottom, PAofInning, PitchofPA) 

D1TM25$row_index <- seq_len(nrow(D1TM25))
  


# One-hot encode TaggedPitchType
D1TM25 <- D1TM25 %>%
  mutate(TaggedPitchType = as.factor(TaggedPitchType))

# Create one-hot encoding matrix for TaggedPitchType
TaggedPitchType_onehot <- model.matrix(~ TaggedPitchType - 1, data = D1TM25)

# Combine one-hot encoded TaggedPitchType with PlateLocHeight and VAA
model_data <- cbind(D1TM25 %>% select(PlateLocSide, VertApprAngle), TaggedPitchType_onehot)

# Fit the linear model
VAA_model <- lm(VertApprAngle ~ PlateLocSide + ., data = model_data)

# Predict expected VAA using the linear model
D1TM25$xVAA <- predict(VAA_model, newdata = model_data)

#===============================================================================
# LIHAA / LIVAA Feature Engineering (CORRECTED)
#===============================================================================

# Ensure TaggedPitchType is a factor
D1TM25 <- D1TM25 %>%
  mutate(TaggedPitchType = as.factor(TaggedPitchType))

# --- VAA Model (Vertical Approach Angle) ---
# Create one-hot encoding WITHOUT intercept
vaa_onehot <- model.matrix(~ TaggedPitchType - 1, data = D1TM25)

# Build model data frame
vaa_data <- data.frame(
  VertApprAngle = D1TM25$VertApprAngle,
  PlateLocSide = D1TM25$PlateLocSide
)
vaa_data <- cbind(vaa_data, vaa_onehot)

# Fit model WITHOUT additional intercept
VAA_model <- lm(VertApprAngle ~ . - 1, data = vaa_data)  # Note the "- 1"

# Predict expected VAA
D1TM25$xVAA <- predict(VAA_model, newdata = vaa_data)

cat("VAA model coefficients:\n")
print(head(names(coef(VAA_model)), 10))

# --- HAA Model (Horizontal Approach Angle) ---
# Create one-hot encoding WITHOUT intercept
haa_onehot <- model.matrix(~ TaggedPitchType - 1, data = D1TM25)

# Build model data frame
haa_data <- data.frame(
  HorzApprAngle = D1TM25$HorzApprAngle,
  PlateLocHeight = D1TM25$PlateLocHeight
)
haa_data <- cbind(haa_data, haa_onehot)

# Fit model WITHOUT additional intercept
HAA_model <- lm(HorzApprAngle ~ . - 1, data = haa_data)  # Note the "- 1"

# Predict expected HAA
D1TM25$xHAA <- predict(HAA_model, newdata = haa_data)

cat("HAA model coefficients:\n")
print(head(names(coef(HAA_model)), 10))

# Calculate LIHAA and LIVAA
D1TM25 <- D1TM25 %>%
  mutate(
    LIVAA = xVAA - VertApprAngle,
    LIHAA = xHAA - HorzApprAngle
  )

cat("\nLIHAA range in training:", range(D1TM25$LIHAA, na.rm = TRUE), "\n")
cat("LIVAA range in training:", range(D1TM25$LIVAA, na.rm = TRUE), "\n")

# Apply the conditions to create the batted_ball_type column
D1TM25 <- D1TM25 %>%
  mutate(batted_ball_type = case_when(
    PitchCall != "InPlay" ~ NA_character_,
    Angle < 10 ~ "GroundBall",
    Angle >= 10 & Angle < 25 ~ "LineDrive",
    Angle >= 25 & Angle < 50 ~ "FlyBall",
    Angle >= 50 ~ "PopUp"
  ))

D1TM25 <- D1TM25 %>%
  mutate(ground_ball = ifelse(batted_ball_type == "GroundBall", 1, 0)
  )

D1TM25 <- D1TM25 %>%
  mutate(line_drive = ifelse(batted_ball_type == "LineDrive", 1, 0)
  )

D1TM25 <- D1TM25 %>%
  mutate(fly_ball = ifelse(batted_ball_type == "FlyBall", 1, 0)
  )

D1TM25 <- D1TM25 %>%
  mutate(pop_up = ifelse(batted_ball_type == "PopUp", 1, 0)
  )

# Remove bunts
D1TM25 <- filter(D1TM25, TaggedHitType != "Bunt")

# Save all trained models
models_path <- "/Users/a13105/Documents/R Projects/Scripts/Pitcher Reports 2026/www/"

# Save LIHAA/LIVAA models (CRITICAL!)
saveRDS(VAA_model, paste0(models_path, "VAA_model.rds"))
saveRDS(HAA_model, paste0(models_path, "HAA_model.rds"))

#===============================================================================
# PITCHES CALLED MODEL
#===============================================================================
# Define the vector of pitch calls
pitch_call <- c("BallCalled", "StrikeCalled", "HitByPitch", "BallinDirt")

# Filter and preprocess the dataset
D1TM25_pitch_call <- filter(D1TM25, PitchCall %in% pitch_call) %>%
  mutate(PitchCall = case_when(
    PitchCall == "BallCalled" ~ 0,
    PitchCall == "StrikeCalled" ~ 1,
    PitchCall == "HitByPitch" ~ 2,
    PitchCall == "BallinDirt" ~ 0
  )) %>%
  filter(!is.na(PitchCall))

# Prepare features and labels
D1TM25_pitch_call <- D1TM25_pitch_call %>%
  select(PlateLocSide, PlateLocHeight, row_index, PitchCall)

# Split data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(D1TM25_pitch_call), 
                        size = floor(0.75 * nrow(D1TM25_pitch_call)))
train_set <- D1TM25_pitch_call[train_indices, ]
test_set <- D1TM25_pitch_call[-train_indices, ]

# Extract labels and features
y_train <- train_set$PitchCall
y_test <- test_set$PitchCall
X_train <- train_set %>% select(-PitchCall, -row_index)
X_test <- test_set %>% select(-PitchCall, -row_index)

# Convert to DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

# Define XGBoost parameters
params <- list(
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = 3,
  eta = 0.1,
  max_depth = 6,
  gamma = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train the model with early stopping
xgb_model_pitchcall <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

# Apply the model to the full D1TM25 dataset
D1TM25_features <- D1TM25 %>%
  select(PlateLocSide, PlateLocHeight, row_index) %>%
  filter(!is.na(PlateLocSide) & !is.na(PlateLocHeight)) # Ensure no missing values in features

dpbp <- xgb.DMatrix(data = as.matrix(D1TM25_features %>% select(-row_index)))

# Predict probabilities
pred_probs <- predict(xgb_model_pitchcall, dpbp)
pred_probs <- matrix(pred_probs, ncol = 3, byrow = TRUE)

# Add predicted probabilities to D1TM25
# Create a data frame with predicted probabilities and their corresponding indices
predicted_probs <- data.frame(
  row_index = D1TM25_features$row_index,
  xBall = pred_probs[, 1],
  xStrike = pred_probs[, 2],
  xHBP = pred_probs[, 3]
)

# Join the predictions back to D1TM25 by row_index
D1TM25 <- D1TM25 %>%
  left_join(predicted_probs, by = "row_index")


rm(D1TM25_pitch_call)
rm(D1TM25_features)
rm(dpbp)
rm(predictions_by_type)
rm(predicted_probs)
rm(train_set)
rm(test_set)
rm(VAA_model)
rm(X_test)
rm(X_train)
rm(xgb_model)
rm(pred_probs)
rm(evaluation_summary)
rm(TaggedPitchType_onehot)

#===============================================================================
# PITCHES SWUNG AT MODEL
#===============================================================================
# Define the vector of swing types
swing_types <- c("StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "InPlay", "FoulBallFieldable")

# Filter the data for the specified pitch calls to create training dataset
filtered_D1TM25_swing <- filter(D1TM25, PitchCall %in% swing_types)

# Create the 'contact' column based on the description for training data
D1TM25_swing <- filtered_D1TM25_swing %>%
  mutate(whiff = if_else(PitchCall %in% c("FoulBall", "InPlay"), 0, 1))

# Prepare features for training
one_hot_D1TM25_swing <- D1TM25_swing %>%
  select(TaggedPitchType, PitcherThrows, Balls, Strikes, PitcherThrows, BatterSide, SpinRate, SpinAxis, RelSpeed, az0, ax0, 
         RelSide, RelHeight, Extension, 
         PlateLocSide, PlateLocHeight, row_index, LIHAA, LIVAA, whiff)

# Transform categorical variables
one_hot_D1TM25_swing <- one_hot_D1TM25_swing %>%
  mutate(
    BatterSide = ifelse(BatterSide == "Right", 1, 0),
    PitcherThrows = ifelse(PitcherThrows == "Right", 1, 0)
  )

# One-hot encode TaggedPitchType
TaggedPitchType_onehot_matrix <- model.matrix(~ TaggedPitchType - 1, one_hot_D1TM25_swing)
one_hot_D1TM25_swing <- cbind(one_hot_D1TM25_swing, TaggedPitchType_onehot_matrix) %>%
  select(-TaggedPitchType)

# Prepare features for full dataset (excluding whiff column)
one_hot_D1TM25_full <- D1TM25 %>%
  select(TaggedPitchType, PitcherThrows, Balls, Strikes, PitcherThrows, BatterSide, SpinRate, SpinAxis, RelSpeed, az0, ax0, 
         RelSide, RelHeight, Extension, 
         PlateLocSide, PlateLocHeight, row_index, LIHAA, LIVAA)

# Transform categorical variables for full dataset
one_hot_D1TM25_full <- one_hot_D1TM25_full %>%
  mutate(
    BatterSide = ifelse(BatterSide == "Right", 1, 0),
    PitcherThrows = ifelse(PitcherThrows == "Right", 1, 0)
  )

# One-hot encode TaggedPitchType for full dataset
TaggedPitchType_onehot_matrix_full <- model.matrix(~ TaggedPitchType - 1, one_hot_D1TM25_full)
one_hot_D1TM25_full <- cbind(one_hot_D1TM25_full, TaggedPitchType_onehot_matrix_full) %>%
  select(-TaggedPitchType)

# Ensure full dataset has the same columns as training dataset
# Identify columns present in training dataset but not in full dataset
training_columns <- colnames(one_hot_D1TM25_swing %>% select(-whiff))
missing_columns <- setdiff(training_columns, colnames(one_hot_D1TM25_full))

# Add missing columns to full dataset with 0 values if necessary
for (col in missing_columns) {
  one_hot_D1TM25_full[[col]] <- 0
}

# Ensure columns are in the same order
one_hot_D1TM25_full <- one_hot_D1TM25_full[, training_columns]

# Prepare training data
X_train <- one_hot_D1TM25_swing %>% select(-whiff)
y_train <- one_hot_D1TM25_swing$whiff

# Convert to matrix format
X_train_matrix <- as.matrix(X_train)
X_full_matrix <- as.matrix(one_hot_D1TM25_full)

# Create LightGBM dataset for training
lgb_train <- lgb.Dataset(data = X_train_matrix, label = y_train)

# Modified Bayesian Optimization Function
whiff_objective <- function(num_leaves, max_depth, learning_rate) {
  # Prepare parameters
  params <- list(
    objective = "binary",
    metric = "binary_logloss",
    num_leaves = round(num_leaves),
    max_depth = round(max_depth),
    learning_rate = learning_rate,
    min_data_in_leaf = 30,  # Fixed value instead of dynamic
    feature_pre_filter = FALSE,  # Add this to allow dynamic parameter changes
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq = 1,
    scale_pos_weight = (length(y_train) - sum(y_train)) / sum(y_train)
  )
  
  # Perform cross-validation
  cv_results <- lgb.cv(
    params = params,
    data = lgb_train,
    nrounds = 1000,
    nfold = 5,
    early_stopping_rounds = 50,
    verbose = -1
  )
  
  # Return best score (minimize binary logloss)
  best_score <- min(cv_results$best_score)
  return(list(Score = -best_score))  # Negative because optimizer maximizes
}

# Updated parameter search bounds
bounds <- list(
  num_leaves = c(20L, 100L),
  max_depth = c(3L, 15L),
  learning_rate = c(0.01, 0.3)
)

# Perform Bayesian Optimization
opt_results <- BayesianOptimization(
  FUN = whiff_objective,
  bounds = bounds,
  init_points = 5,
  n_iter = 10,
  acq = "ucb"
)

# Extract best parameters
best_params <- list(
  objective = "binary",
  metric = "binary_logloss",
  num_leaves = round(opt_results$Best_Par["num_leaves"]),
  max_depth = round(opt_results$Best_Par["max_depth"]),
  learning_rate = opt_results$Best_Par["learning_rate"],
  min_data_in_leaf = 30,  # Fixed value
  feature_pre_filter = FALSE,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 1,
  scale_pos_weight = (length(y_train) - sum(y_train)) / sum(y_train)
)

# Train final model with optimized parameters
final_model_swings <- lgb.train(
  params = best_params,
  data = lgb_train,
  nrounds = 1000,
  verbose = 0
)

# Make predictions on full dataset
pred_probs <- predict(final_model_swings, X_full_matrix)

# Add predictions to the original dataset
D1TM25$xWhiff <- pred_probs

# Optional: ROC Curve for Training Data
# First, create a test set from the training data
set.seed(123)
sample_split <- sample.split(Y = y_train, SplitRatio = 0.75)
train_subset <- subset(one_hot_D1TM25_swing, sample_split == TRUE)
test_subset <- subset(one_hot_D1TM25_swing, sample_split == FALSE)

X_test_subset <- test_subset %>% select(-whiff)
y_test_subset <- test_subset$whiff
X_test_matrix_subset <- as.matrix(X_test_subset)

# Predictions on test subset
test_pred_probs <- predict(final_model_swings, as.matrix(X_test_subset))

# ROC Curve
roc_curve <- roc(y_test_subset, test_pred_probs)
auc_value <- auc(roc_curve)

# Plot ROC Curve
roc_data <- data.frame(
  tpr = rev(roc_curve$sensitivities),
  fpr = rev(1 - roc_curve$specificities)
)

ggplot(roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(
    title = "ROC Curve for Whiff Prediction Model",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  annotate("text", x = 0.7, y = 0.2, 
           label = paste("AUC =", round(auc_value, 4)), 
           color = "red") +
  theme_minimal()

# Feature Importance Plot
importance_matrix <- lgb.importance(final_model_swings, percentage = TRUE)
importance_matrix <- importance_matrix %>%
  rename(Feature = Feature, Importance = Gain)

ggplot(data = importance_matrix, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(
    title = "Feature Importance",
    x = "Features",
    y = "Importance (Gain)"
  ) +
  theme_minimal()

rm(D1TM25_swing)
rm(filtered_D1TM25_swing)
rm(one_hot_D1TM25_swing)
rm(TaggedPitchType_onehot_matrix)
rm(one_hot_D1TM25_full)
rm(training_columns)
rm(missing_columns)
rm(X_train)
rm(y_train)
rm(X_train_matrix)
rm(X_full_matrix)
rm(train_subset)
rm(test_subset)
rm(X_test_subset)
rm(y_test_subset)
rm(X_test_matrix_subset)
rm(test_pred_probs)
rm(pred_probs)
rm(roc_data)
rm(roc_curve)




#===============================================================================
# PITCHES MADE CONTACT WITH MODEL
#===============================================================================
# Define the vector of pitch calls
contact_types <- c("FoulBall", "FoulBallNotFieldable", "InPlay", "FoulBallFieldable")

# Filter the data for the specified pitch calls to create training dataset
filtered_D1TM25_contact <- filter(D1TM25, PitchCall %in% contact_types)

# Create the 'contact' column based on the description for training data
D1TM25_contact <- filtered_D1TM25_contact %>%
  mutate(in_play = if_else(PitchCall %in% c("Foul", "FoulBallNotFieldable", "FoulBallFieldable"), 0, 1))

# Prepare features for training
one_hot_D1TM25_contact <- D1TM25_contact %>%
  select(Balls, Strikes, PitcherThrows, BatterSide, SpinRate, SpinAxis, RelSpeed, az0, ax0, 
         RelSide, RelHeight, Extension, 
         PlateLocSide, PlateLocHeight, row_index, LIHAA, LIVAA, in_play)

# Transform categorical variables
one_hot_D1TM25_contact <- one_hot_D1TM25_contact %>%
  mutate(
    BatterSide = ifelse(BatterSide == "Right", 1, 0),
    PitcherThrows = ifelse(PitcherThrows == "Right", 1, 0)
  )

# Prepare features for full dataset (excluding in_play column)
one_hot_D1TM25_full <- D1TM25 %>%
  select(Balls, Strikes, PitcherThrows, BatterSide, SpinRate, SpinAxis, RelSpeed, az0, ax0, 
         RelSide, RelHeight, Extension, 
         PlateLocSide, PlateLocHeight, row_index, LIHAA, LIVAA)

# Transform categorical variables for full dataset
one_hot_D1TM25_full <- one_hot_D1TM25_full %>%
  mutate(
    BatterSide = ifelse(BatterSide == "Right", 1, 0),
    PitcherThrows = ifelse(PitcherThrows == "Right", 1, 0)
  )

# Ensure full dataset has the same columns as training dataset
# Identify columns present in training dataset but not in full dataset
training_columns <- colnames(one_hot_D1TM25_contact %>% select(-in_play))
missing_columns <- setdiff(training_columns, colnames(one_hot_D1TM25_full))

# Add missing columns to full dataset with 0 values if necessary
for (col in missing_columns) {
  one_hot_D1TM25_full[[col]] <- 0
}

# Ensure columns are in the same order
one_hot_D1TM25_full <- one_hot_D1TM25_full[, training_columns]

# Prepare training data
X_train <- one_hot_D1TM25_contact %>% select(-in_play)
y_train <- one_hot_D1TM25_contact$in_play

# Convert to matrix format
X_train_matrix <- as.matrix(X_train)
X_full_matrix <- as.matrix(one_hot_D1TM25_full)

# Create LightGBM dataset for training
lgb_train <- lgb.Dataset(data = X_train_matrix, label = y_train)

# Define the Objective Function for Bayesian Optimization
lgb_opt_function <- function(num_leaves, max_depth, learning_rate, feature_fraction, bagging_fraction) {
  
  params <- list(
    objective = "binary",
    metric = "binary_logloss",
    num_leaves = round(num_leaves),
    max_depth = round(max_depth),
    learning_rate = learning_rate,
    feature_fraction = feature_fraction,
    bagging_fraction = bagging_fraction,
    bagging_freq = 1,
    min_data_in_leaf = 20,
    scale_pos_weight = (length(y_train) - sum(y_train)) / sum(y_train)
  )
  
  lgb_cv <- lgb.cv(
    params = params,
    data = lgb_train,
    nrounds = 100,
    nfold = 5,
    verbose = -1,
    stratified = TRUE,
    early_stopping_rounds = 10
  )
  
  # Extract the minimum binary_logloss value
  binary_logloss_vals <- unlist(lgb_cv$record_evals$valid$binary_logloss$eval)
  min_logloss <- min(binary_logloss_vals, na.rm = TRUE)
  
  list(Score = -min_logloss, Pred = NA)
}

# Perform Bayesian Optimization
opt_results <- BayesianOptimization(
  FUN = lgb_opt_function,
  bounds = list(
    num_leaves = c(20L, 50L),
    max_depth = c(5L, 15L),
    learning_rate = c(0.01, 0.2),
    feature_fraction = c(0.6, 1.0),
    bagging_fraction = c(0.6, 1.0)
  ),
  init_points = 5,
  n_iter = 10,
  acq = "ucb",
  kappa = 2.576,
  verbose = TRUE
)

# Get the best parameters from the Bayesian Optimization result
best_params <- opt_results$Best_Par

# Extract the parameters as a flat list
best_params <- unlist(best_params)

# Ensure the parameters are in the correct format for LightGBM
params <- list(
  objective = "binary",
  metric = "binary_logloss",
  num_leaves = round(best_params['num_leaves']),
  max_depth = round(best_params['max_depth']),
  learning_rate = best_params['learning_rate'],
  feature_fraction = best_params['feature_fraction'],
  bagging_fraction = best_params['bagging_fraction'],
  bagging_freq = 1,
  min_data_in_leaf = 20,
  scale_pos_weight = (length(y_train) - sum(y_train)) / sum(y_train)
)

# Train the final model using the optimized parameters
final_model_contact <- lgb.train(
  params = params,
  data = lgb_train,
  nrounds = 1000,
  verbose = 0
)

# Make predictions on the full dataset
pred_probs <- predict(final_model_contact, X_full_matrix)

# Add predictions to the original dataset
D1TM25$xIn_Play <- pred_probs


# Optional: ROC Curve for Training Data
# First, create a test set from the training data
set.seed(123)
sample_split <- sample.split(Y = y_train, SplitRatio = 0.75)
train_subset <- subset(one_hot_D1TM25_contact, sample_split == TRUE)
test_subset <- subset(one_hot_D1TM25_contact, sample_split == FALSE)

X_test_subset <- test_subset %>% select(-in_play)
y_test_subset <- test_subset$in_play
X_test_matrix_subset <- as.matrix(X_test_subset)

# Predictions on test subset
test_pred_probs <- predict(final_model_contact, X_test_matrix_subset)

# ROC Curve
roc_curve <- roc(y_test_subset, test_pred_probs)
auc_value <- auc(roc_curve)

# Plot ROC Curve
roc_data <- data.frame(
  tpr = rev(roc_curve$sensitivities),
  fpr = rev(1 - roc_curve$specificities)
)

ggplot(roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(
    title = "ROC Curve for Contact Type Prediction Model",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  annotate("text", x = 0.7, y = 0.2, 
           label = paste("AUC =", round(auc_value, 4)), 
           color = "red") +
  theme_minimal()

# Feature Importance Plot
importance_matrix <- lgb.importance(final_model_contact, percentage = TRUE)
importance_matrix <- importance_matrix %>%
  rename(Feature = Feature, Importance = Gain)

ggplot(data = importance_matrix, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(
    title = "Feature Importance",
    x = "Features",
    y = "Importance (Gain)"
  ) +
  theme_minimal()

rm(D1TM25_contact)
rm(filtered_D1TM25_contact)
rm(one_hot_D1TM25_contact)
rm(TaggedPitchType_onehot_matrix)
rm(one_hot_D1TM25_full)
rm(training_columns)
rm(missing_columns)
rm(X_train)
rm(y_train)
rm(X_train_matrix)
rm(X_full_matrix)
rm(train_subset)
rm(test_subset)
rm(X_test_subset)
rm(y_test_subset)
rm(X_test_matrix_subset)
rm(test_pred_probs)
rm(pred_probs)
rm(roc_data)
rm(roc_curve)
rm(lgb_train)

#===============================================================================
# HIT TYPE PREDICTION MODEL
#===============================================================================
library(dplyr)
library(lightgbm)
library(rBayesianOptimization)
library(caTools)
library(pROC)
library(ggplot2)

# Define target columns
target_columns <- c("ground_ball", "line_drive", "fly_ball", "pop_up")

# Remove rows with NA in ground_ball column and filter in-play hits for training dataset
D1TM25_in_play <- D1TM25 %>%
  filter(PitchCall == "InPlay") %>%
  filter(!is.na(ground_ball))

# Data Preparation:
# Remove rows with NA in target columns and select relevant features
D1TM25_in_play <- D1TM25_in_play %>%
  filter(!is.na(ground_ball) & !is.na(line_drive) & !is.na(fly_ball) & !is.na(pop_up))
relevant_features <- c("TaggedPitchType", "PitcherThrows", "Balls", "Strikes", "BatterSide",
                       "RelSpeed", "az0", "ax0", "RelSide", "RelHeight", "Extension", 
                       "PlateLocSide", "PlateLocHeight", "SpinRate", "SpinAxis", 
                       "LIHAA", "LIVAA", "row_index")

# Prepare dataset for training
data_prep <- D1TM25_in_play %>%
  select(all_of(relevant_features), ground_ball, line_drive, fly_ball, pop_up) %>%
  mutate(
    PitcherThrows = ifelse(PitcherThrows == "Right", 1, 0),
    BatterSide = ifelse(BatterSide == "Right", 1, 0)  # Encode categorical variables
  )

# Prepare dataset for prediction (entire dataset)
prediction_data_prep <- D1TM25 %>%
  select(all_of(relevant_features)) %>%
  mutate(
    PitcherThrows = ifelse(PitcherThrows == "Right", 1, 0),
    BatterSide = ifelse(BatterSide == "Right", 1, 0)  # Encode categorical variables
  )

# One-hot encode TaggedPitchType for training data
TaggedPitchType_onehot_matrix_train <- model.matrix(~ TaggedPitchType - 1, data_prep)
data_prep <- cbind(data_prep, TaggedPitchType_onehot_matrix_train) %>%
  select(-TaggedPitchType)

# One-hot encode TaggedPitchType for prediction data
TaggedPitchType_onehot_matrix_prediction <- model.matrix(~ TaggedPitchType - 1, prediction_data_prep)
prediction_data_prep <- cbind(prediction_data_prep, TaggedPitchType_onehot_matrix_prediction) %>%
  select(-TaggedPitchType)

# Ensure prediction data has the same columns as training data
missing_cols <- setdiff(names(data_prep)[!names(data_prep) %in% target_columns], names(prediction_data_prep))
for (col in missing_cols) {
  prediction_data_prep[[col]] <- 0
}

# Reorder columns to match training data
prediction_data_prep <- prediction_data_prep[, names(data_prep)[!names(data_prep) %in% target_columns]]

# Split training data into train/test sets
set.seed(123)
sample_split <- sample.split(Y = data_prep$ground_ball, SplitRatio = 0.75)
train_set <- subset(data_prep, sample_split == TRUE)
test_set <- subset(data_prep, sample_split == FALSE)

# Prepare for model training
X_train <- train_set %>% select(-all_of(target_columns))
X_test <- test_set %>% select(-all_of(target_columns))

# Updated train_model function with extensive error checking
train_model <- function(target, X_train, X_test, train_set, test_set) {
  # Check for valid data
  if (nrow(X_train) == 0 || nrow(train_set) == 0) {
    stop(paste("No training data for", target))
  }
  
  y_train <- train_set[[target]]
  y_test <- test_set[[target]]
  
  # Validate target variable
  if (length(unique(y_train)) < 2) {
    stop(paste("Insufficient variation in target variable", target))
  }
  
  # LightGBM Dataset preparation
  lgb_train <- lgb.Dataset(data = as.matrix(X_train), label = y_train)
  lgb_test <- lgb.Dataset(data = as.matrix(X_test), label = y_test, reference = lgb_train)
  
  # Objective function for Bayesian Optimization
  objective_function <- function(num_leaves, max_depth, learning_rate, min_data_in_leaf) {
    params <- list(
      objective = "binary",
      metric = "binary_logloss",
      num_leaves = round(num_leaves),
      learning_rate = learning_rate,
      max_depth = round(max_depth),
      min_data_in_leaf = round(min_data_in_leaf),
      feature_fraction = 0.8,
      bagging_fraction = 0.8,
      bagging_freq = 1,
      scale_pos_weight = (length(y_train) - sum(y_train)) / sum(y_train),
      feature_pre_filter = FALSE
    )
    
    tryCatch({
      model <- lgb.train(
        params = params,
        data = lgb_train,
        nrounds = 1000,
        verbose = -1
      )
      
      pred_probs <- predict(model, as.matrix(X_test))
      roc_curve <- roc(y_test, pred_probs)
      auc_value <- auc(roc_curve)
      
      return(list(Score = auc_value))
    }, error = function(e) {
      cat("Error in optimization for", target, ":", e$message, "\n")
      return(list(Score = 0))
    })
  }
  
  # Bayesian Optimization
  bounds <- list(
    num_leaves = c(20L, 50L),
    max_depth = c(5L, 15L),
    learning_rate = c(0.01, 0.2),
    min_data_in_leaf = c(10L, 30L)
  )
  
  opt_results <- tryCatch({
    BayesianOptimization(
      FUN = objective_function,
      bounds = bounds,
      init_points = 5,
      n_iter = 10,
      acq = "ucb"
    )
  }, error = function(e) {
    cat("Bayesian Optimization failed for", target, ":", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(opt_results)) {
    stop(paste("Optimization failed for", target))
  }
  
  # Train final model using best parameters
  best_params <- list(
    objective = "binary",
    metric = "binary_logloss",
    num_leaves = opt_results$Best_Par[["num_leaves"]],
    learning_rate = opt_results$Best_Par[["learning_rate"]],
    max_depth = opt_results$Best_Par[["max_depth"]],
    min_data_in_leaf = opt_results$Best_Par[["min_data_in_leaf"]],
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq = 1,
    scale_pos_weight = (length(y_train) - sum(y_train)) / sum(y_train),
    feature_pre_filter = FALSE
  )
  
  final_model_in_play <- tryCatch({
    lgb.train(
      params = best_params,
      data = lgb_train,
      nrounds = 1000,
      verbose = -1
    )
  }, error = function(e) {
    cat("Model training failed for", target, ":", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(final_model_in_play)) {
    stop(paste("Failed to train model for", target))
  }
  
  # Predictions and Evaluation
  pred_probs <- predict(final_model_in_play, as.matrix(X_test))
  pred_classes <- ifelse(pred_probs > 0.5, 1, 0)
  
  auc_value <- auc(roc(y_test, pred_probs))
  accuracy <- sum(pred_classes == y_test) / length(y_test)
  
  assign(paste0("final_model_in_play_", target), final_model_in_play, envir = .GlobalEnv)
  
  return(list(
    model = final_model_in_play,
    auc = auc_value,
    accuracy = accuracy,
    predictions = list(test_probs = pred_probs, test_classes = pred_classes)
  ))
}

# Train models for all target variables and store results
results <- list()
for (target in target_columns) {
  cat("Training model for:", target, "\n")
  results[[target]] <- train_model(target, X_train, X_test, train_set, test_set)
}

# Predict probabilities for the entire dataset
predictions <- list()
for (target in target_columns) {
  cat("Predicting for:", target, "\n")
  predictions[[target]] <- predict(results[[target]]$model, as.matrix(prediction_data_prep))
}

# Add predictions to the original D1TM25 dataset using vectorized assignment
D1TM25$xGB <- predictions$ground_ball
D1TM25$xLD <- predictions$line_drive
D1TM25$xFB <- predictions$fly_ball
D1TM25$xPU <- predictions$pop_up

# Perform vectorized normalization (each row's sum is computed and used to normalize the predictions)
D1TM25 <- D1TM25 %>%
  mutate(
    sum_pred = xGB + xLD + xFB + xPU,
    xGB = xGB / sum_pred,
    xLD = xLD / sum_pred,
    xFB = xFB / sum_pred,
    xPU = xPU / sum_pred
  ) %>%
  select(-sum_pred)

# Evaluation Summary
evaluation_summary <- lapply(results, function(res) {
  list(auc = res$auc, accuracy = res$accuracy)
})
print(evaluation_summary)

# Visualization: ROC Curve for ground_ball
roc_curve <- roc(test_set$ground_ball, results$ground_ball$predictions$test_probs)
ggplot(data.frame(tpr = rev(roc_curve$sensitivities), fpr = rev(roc_curve$specificities)), 
       aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(
    title = "ROC Curve for Hit Type Prediction",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  annotate("text", x = 0.7, y = 0.2, label = paste("AUC =", round(roc_curve$auc, 4)), color = "red") +
  theme_minimal()

# Clean up unnecessary objects from workspace
rm(D1TM25_in_play)
rm(data_prep)
rm(TaggedPitchType_onehot_matrix_train)
rm(TaggedPitchType_onehot_matrix_prediction)
rm(prediction_data_prep)
rm(X_train)
rm(X_test)
rm(train_set)
rm(test_set)
#===============================================================================
# SWING PREDICTION MODEL
#===============================================================================
# Create a new column `swing` based on the `description` column
D1TM25$swing <- ifelse(
  D1TM25$PitchCall %in% c("InPlay", "FoulBall", 
                                   "StrikeSwinging", "FoulBallNotFieldable", "FoulBallFieldable"),
  1,
  0
)


# Prepare features for training
one_hot_D1TM25 <- D1TM25 %>%
  select(TaggedPitchType, PitcherThrows, Balls, Strikes, PitcherThrows, BatterSide, SpinRate, SpinAxis, RelSpeed, az0, ax0, 
         RelSide, RelHeight, Extension, 
         PlateLocSide, PlateLocHeight, row_index, LIHAA, LIVAA, swing)

# Transform categorical variables
one_hot_D1TM25 <- one_hot_D1TM25 %>%
  mutate(
    BatterSide = ifelse(BatterSide == "Right", 1, 0),
    PitcherThrows = ifelse(PitcherThrows == "Right", 1, 0)
  )

# One-hot encode TaggedPitchType
TaggedPitchType_onehot_matrix <- model.matrix(~ TaggedPitchType - 1, one_hot_D1TM25)
one_hot_D1TM25 <- cbind(one_hot_D1TM25, TaggedPitchType_onehot_matrix) %>%
  select(-TaggedPitchType)

# Prepare features for full dataset (excluding whiff column)
one_hot_D1TM25_full <- D1TM25 %>%
  select(TaggedPitchType, PitcherThrows, Balls, Strikes, PitcherThrows, BatterSide, SpinRate, SpinAxis, RelSpeed, az0, ax0, 
         RelSide, RelHeight, Extension, 
         PlateLocSide, PlateLocHeight, row_index, LIHAA, LIVAA)

# Transform categorical variables for full dataset
one_hot_D1TM25_full <- one_hot_D1TM25_full %>%
  mutate(
    BatterSide = ifelse(BatterSide == "Right", 1, 0),
    PitcherThrows = ifelse(PitcherThrows == "Right", 1, 0)
  )

# One-hot encode TaggedPitchType for full dataset
TaggedPitchType_onehot_matrix_full <- model.matrix(~ TaggedPitchType - 1, one_hot_D1TM25_full)
one_hot_D1TM25_full <- cbind(one_hot_D1TM25_full, TaggedPitchType_onehot_matrix_full) %>%
  select(-TaggedPitchType)

# Ensure full dataset has the same columns as training dataset
# Identify columns present in training dataset but not in full dataset
training_columns <- colnames(one_hot_D1TM25 %>% select(-swing))
missing_columns <- setdiff(training_columns, colnames(one_hot_D1TM25_full))

# Add missing columns to full dataset with 0 values if necessary
for (col in missing_columns) {
  one_hot_D1TM25_full[[col]] <- 0
}

# Ensure columns are in the same order
one_hot_D1TM25_full <- one_hot_D1TM25_full[, training_columns]

# Prepare training data
X_train <- one_hot_D1TM25 %>% select(-swing)
y_train <- one_hot_D1TM25$swing

# Convert to matrix format
X_train_matrix <- as.matrix(X_train)
X_full_matrix <- as.matrix(one_hot_D1TM25_full)

# Create LightGBM dataset for training
lgb_train <- lgb.Dataset(data = X_train_matrix, label = y_train)

# Modified Bayesian Optimization Function
whiff_objective <- function(num_leaves, max_depth, learning_rate) {
  # Prepare parameters
  params <- list(
    objective = "binary",
    metric = "binary_logloss",
    num_leaves = round(num_leaves),
    max_depth = round(max_depth),
    learning_rate = learning_rate,
    min_data_in_leaf = 30,  # Fixed value instead of dynamic
    feature_pre_filter = FALSE,  # Add this to allow dynamic parameter changes
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq = 1,
    scale_pos_weight = (length(y_train) - sum(y_train)) / sum(y_train)
  )
  
  # Perform cross-validation
  cv_results <- lgb.cv(
    params = params,
    data = lgb_train,
    nrounds = 1000,
    nfold = 5,
    early_stopping_rounds = 50,
    verbose = -1
  )
  
  # Return best score (minimize binary logloss)
  best_score <- min(cv_results$best_score)
  return(list(Score = -best_score))  # Negative because optimizer maximizes
}

# Updated parameter search bounds
bounds <- list(
  num_leaves = c(20L, 100L),
  max_depth = c(3L, 15L),
  learning_rate = c(0.01, 0.3)
)

# Perform Bayesian Optimization
opt_results <- BayesianOptimization(
  FUN = whiff_objective,
  bounds = bounds,
  init_points = 5,
  n_iter = 10,
  acq = "ucb"
)

# Extract best parameters
best_params <- list(
  objective = "binary",
  metric = "binary_logloss",
  num_leaves = round(opt_results$Best_Par["num_leaves"]),
  max_depth = round(opt_results$Best_Par["max_depth"]),
  learning_rate = opt_results$Best_Par["learning_rate"],
  min_data_in_leaf = 30,  # Fixed value
  feature_pre_filter = FALSE,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 1,
  scale_pos_weight = (length(y_train) - sum(y_train)) / sum(y_train)
)

# Train final model with optimized parameters
final_model_xSwing <- lgb.train(
  params = best_params,
  data = lgb_train,
  nrounds = 1000,
  verbose = 0
)

# Make predictions on full dataset
pred_probs <- predict(final_model_xSwing, X_full_matrix)

# Add predictions to the original dataset
D1TM25$xSwing <- pred_probs

# Optional: ROC Curve for Training Data
# First, create a test set from the training data
set.seed(123)
sample_split <- sample.split(Y = y_train, SplitRatio = 0.75)
train_subset <- subset(one_hot_D1TM25, sample_split == TRUE)
test_subset <- subset(one_hot_D1TM25, sample_split == FALSE)

X_test_subset <- test_subset %>% select(-swing)
y_test_subset <- test_subset$swing
X_test_matrix_subset <- as.matrix(X_test_subset)

# Predictions on test subset
test_pred_probs <- predict(final_model_xSwing, as.matrix(X_test_subset))

# ROC Curve
roc_curve <- roc(y_test_subset, test_pred_probs)
auc_value <- auc(roc_curve)

# Plot ROC Curve
roc_data <- data.frame(
  tpr = rev(roc_curve$sensitivities),
  fpr = rev(1 - roc_curve$specificities)
)

ggplot(roc_data, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(
    title = "ROC Curve for Swing Prediction Model",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  annotate("text", x = 0.7, y = 0.2, 
           label = paste("AUC =", round(auc_value, 4)), 
           color = "red") +
  theme_minimal()

# Feature Importance Plot
importance_matrix <- lgb.importance(final_model_xSwing, percentage = TRUE)
importance_matrix <- importance_matrix %>%
  rename(Feature = Feature, Importance = Gain)

ggplot(data = importance_matrix, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(
    title = "Feature Importance",
    x = "Features",
    y = "Importance (Gain)"
  ) +
  theme_minimal()

#===============================================================================
# Find most likely expected event
#===============================================================================

# --- Replace this line ---
# D1TM25 <- D1TM25 %>% mutate(xSwing = 1)

# --- With this: create a hard decision from xSwing ---
D1TM25 <- D1TM25 %>%
  mutate(predicted_swing = if_else(xSwing >= 0.5, 1L, 0L))


D1TM25$xContact <- 1 - D1TM25$xWhiff
D1TM25$xFoul    <- 1 - D1TM25$xIn_Play

# Ensure D1TM25 is a data frame
D1TM25 <- as.data.frame(D1TM25)

# Probabilities at each leaf using predicted_swing
D1TM25 <- D1TM25 %>%
  mutate(
    # Swings (only if predicted_swing == 1)
    pct_Whiff               = predicted_swing * xWhiff              * 100,
    pct_Contact_Foul        = predicted_swing * xContact * xFoul    * 100,
    pct_Contact_InPlay_GB   = predicted_swing * xContact * xIn_Play * xGB * 100,
    pct_Contact_InPlay_LD   = predicted_swing * xContact * xIn_Play * xLD * 100,
    pct_Contact_InPlay_FB   = predicted_swing * xContact * xIn_Play * xFB * 100,
    pct_Contact_InPlay_PU   = predicted_swing * xContact * xIn_Play * xPU * 100
  )

# Called outcomes occur only if predicted_swing == 0
D1TM25 <- D1TM25 %>%
  mutate(
    pct_CalledStrike = (1 - predicted_swing) * xStrike * 100,
    pct_Ball         = (1 - predicted_swing) * xBall   * 100,
    pct_HBP          = (1 - predicted_swing) * xHBP    * 100,
    total_pct = pct_CalledStrike + pct_Ball + pct_HBP +
      pct_Whiff + pct_Contact_Foul + 
      pct_Contact_InPlay_GB + pct_Contact_InPlay_LD + 
      pct_Contact_InPlay_FB + pct_Contact_InPlay_PU
  )

# Normalize hit-type probs (unchanged)
D1TM25 <- D1TM25 %>%
  mutate(
    total = xGB + xLD + xFB + xPU,
    xGB = xGB / total,
    xLD = xLD / total,
    xFB = xFB / total,
    xPU = xPU / total
  )

# Argmax to pick expected_event (unchanged except it now uses pct_* built with predicted_swing)
D1TM25 <- D1TM25 %>%
  mutate(
    expected_event = case_when(
      pct_Contact_InPlay_GB == pmax(pct_Contact_InPlay_GB, pct_Contact_InPlay_LD, pct_Contact_InPlay_FB, pct_Contact_InPlay_PU, pct_CalledStrike, pct_Ball, pct_HBP, pct_Whiff, pct_Contact_Foul) ~ "GroundBall",
      pct_Contact_InPlay_LD == pmax(pct_Contact_InPlay_GB, pct_Contact_InPlay_LD, pct_Contact_InPlay_FB, pct_Contact_InPlay_PU, pct_CalledStrike, pct_Ball, pct_HBP, pct_Whiff, pct_Contact_Foul) ~ "LineDrive",
      pct_Contact_InPlay_FB == pmax(pct_Contact_InPlay_GB, pct_Contact_InPlay_LD, pct_Contact_InPlay_FB, pct_Contact_InPlay_PU, pct_CalledStrike, pct_Ball, pct_HBP, pct_Whiff, pct_Contact_Foul) ~ "FlyBall",
      pct_Contact_InPlay_PU == pmax(pct_Contact_InPlay_GB, pct_Contact_InPlay_LD, pct_Contact_InPlay_FB, pct_Contact_InPlay_PU, pct_CalledStrike, pct_Ball, pct_HBP, pct_Whiff, pct_Contact_Foul) ~ "PopUp",
      pct_CalledStrike       == pmax(pct_Contact_InPlay_GB, pct_Contact_InPlay_LD, pct_Contact_InPlay_FB, pct_Contact_InPlay_PU, pct_CalledStrike, pct_Ball, pct_HBP, pct_Whiff, pct_Contact_Foul) ~ "CalledStrike",
      pct_Ball               == pmax(pct_Contact_InPlay_GB, pct_Contact_InPlay_LD, pct_Contact_InPlay_FB, pct_Contact_InPlay_PU, pct_CalledStrike, pct_Ball, pct_HBP, pct_Whiff, pct_Contact_Foul) ~ "CalledBall",
      pct_HBP                == pmax(pct_Contact_InPlay_GB, pct_Contact_InPlay_LD, pct_Contact_InPlay_FB, pct_Contact_InPlay_PU, pct_CalledStrike, pct_Ball, pct_HBP, pct_Whiff, pct_Contact_Foul) ~ "HitByPitch",
      pct_Whiff              == pmax(pct_Contact_InPlay_GB, pct_Contact_InPlay_LD, pct_Contact_InPlay_FB, pct_Contact_InPlay_PU, pct_CalledStrike, pct_Ball, pct_HBP, pct_Whiff, pct_Contact_Foul) ~ "Whiff",
      pct_Contact_Foul       == pmax(pct_Contact_InPlay_GB, pct_Contact_InPlay_LD, pct_Contact_InPlay_FB, pct_Contact_InPlay_PU, pct_CalledStrike, pct_Ball, pct_HBP, pct_Whiff, pct_Contact_Foul) ~ "Foul",
      TRUE ~ NA_character_
    )
  )



# Identify the most likely event
D1TM25 <- D1TM25 %>%
  mutate(
    actual_event = case_when(
      PitchCall %in% c("BallCalled", "BallInDirt") ~ "CalledBall",
      PitchCall == "StrikeCalled" ~ "CalledStrike",
      PitchCall %in% c("FoulBall", "FoulBallFieldable", "FoulBallNotFieldable") ~ "Foul",
      PitchCall == "StrikeSwinging" ~ "Whiff",
      PitchCall == "HitByPitch" ~ "HitByPitch",
      batted_ball_type == "GroundBall" ~ "GroundBall",
      batted_ball_type == "LineDrive" ~ "LineDrive",
      batted_ball_type == "FlyBall" ~ "FlyBall",
      batted_ball_type == "PopUp" ~ "PopUp",
      TRUE ~ NA_character_
    )
  )

# Calculate accuracy
accuracy <- D1TM25 %>%
  summarise(
    total_events = n(),
    correct_predictions = sum(actual_event == expected_event, na.rm = TRUE),
    accuracy = correct_predictions / total_events
  )

# View the result
accuracy


#===============================================================================
# Get run values for events
#===============================================================================

# Define the run values for each count
run_values <- c(
  "0-0" = 0.001695997, "1-0" = 0.039248042, "0-1" = -0.043581338,
  "2-0" = -0.043581338, "1-1" = -0.015277684, "0-2" = -0.103242476,
  "3-0" = 0.200960731, "2-1" = 0.034545018, "1-2" = -0.080485991,
  "3-1" = 0.138254876, "2-2" = -0.039716495, "3-2" = 0.048505049,
  "Walk" = 0.325, "HitByPitch" = 0.325, "Strikeout" = -0.284,
  "FlyBall" = 0.586, "LineDrive" = 0.528,
  "GroundBall" = 0.164, "PopUp" = 0.0186
)

# Calculate expected run value change
D1TM25 <- D1TM25 %>%
  mutate(
    # Current count as a string
    current_count = paste(Balls, Strikes, sep = "-"),
    current_run_value = run_values[current_count],
    
    # New count based on expected event
    new_count = case_when(
      expected_event == "CalledBall" & Balls < 3 ~ paste(Balls + 1, Strikes, sep = "-"),
      expected_event %in% c("CalledStrike", "Whiff", "Foul") & Strikes < 2 ~ paste(Balls, Strikes + 1, sep = "-"),
      expected_event == "Walk" ~ "Walk",
      expected_event == "Strikeout" ~ "Strikeout",
      expected_event %in% c("FlyBall", "LineDrive", "GroundBall", "PopUp") ~ expected_event,
      TRUE ~ current_count # Default to current count if no change
    ),
    new_run_value = run_values[new_count],
    
    # Calculate the run value change
    run_value_change = new_run_value - current_run_value
    
  )

# Calculate actual run value change
D1TM25 <- D1TM25 %>%
  mutate(
    # Current count as a string
    current_count = paste(Balls, Strikes, sep = "-"),
    current_run_value = run_values[current_count],
    
    # New count based on actual event
    new_count_actual = case_when(
      actual_event == "CalledBall" & Balls < 3 ~ paste(Balls + 1, Strikes, sep = "-"),
      actual_event %in% c("CalledStrike", "Whiff", "Foul") & Strikes < 2 ~ paste(Balls, Strikes + 1, sep = "-"),
      actual_event == "Walk" ~ "Walk",
      actual_event == "Strikeout" ~ "Strikeout",
      actual_event %in% c("FlyBall", "LineDrive", "GroundBall", "PopUp") ~ actual_event,
      TRUE ~ current_count # Default to current count if no change
    ),
    new_run_value_actual = run_values[new_count_actual],
    
    # Calculate the run value change
    run_value_change_actual = new_run_value_actual - current_run_value
  )

D1TM25$swing_decision_rv <- D1TM25$run_value_change_actual - D1TM25$run_value_change

D1TM25 <- D1TM25 %>%
  select(-run_value_change_actual, -new_run_value_actual, -new_count_actual, -run_value_change, -new_run_value, -new_count, -current_run_value, -current_count, -total, - total_pct, -xFoul, -xContact, -xPU, -xFB, -xLD, -xGB, -xIn_Play, -xWhiff)

rm(accuracy)
rm(best_params)
rm(bounds)
rm(evaluation_summary)
rm(final_model)
rm(HAA_model)
rm(importance_matrix)
rm(lgb_train)
rm(model_data)
rm(one_hot_D1TM25)
rm(one_hot_D1TM25_full)
rm(opt_results)
rm(params)
rm(predictions)
rm(results)
rm(roc_data)
rm(roc_curve)
rm(TaggedPitchType_onehot_matrix)
rm(TaggedPitchType_onehot_matrix_full)
rm(TaggedPitchType_onehot_matrix_prediction)
rm(TaggedPitchType_onehot_matrix_train)
rm(test_set)
rm(test_subset)
rm(train_set)
rm(train_subset)
rm(X_full_matrix)
rm(X_test)
rm(X_test_matrix_subset)
rm(X_test_subset)
rm(X_train)
rm(X_train_matrix)
rm(auc_value)
rm(col)
rm(contact_types)
rm(dtest)
rm(dtrain)
rm(missing_cols)
rm(missing_columns)
rm(pitch_call)
rm(pred_probs)
rm(relevant_features)
rm(run_values)
rm(sample_split)
rm(swing_types)
rm(target)
rm(target_columns)
rm(test_pred_probs)
rm(train_indices)
rm(training_columns)
rm(y_test)
rm(y_test_subset)
rm(y_train)
rm(lgb_opt_function)
rm(train_model)
rm(whiff_objective)
