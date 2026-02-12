# ==============================================================================
# PROJECT: Billboard Hot 100 - Acoustic Longevity Analysis
# DESCRIPTION: Predicting song long-term success (>1 year) using Logistic
#              Regression and Random Forest models.
# ==============================================================================

# 1. LOAD REQUIRED LIBRARIES ----------------------------------------------------
library(tidyverse)
library(scales)
library(randomForest)
library(MASS)
library(conflicted)
library(car)   # for VIF later
library(pROC)

conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")
conflict_prefer("lag",    "dplyr")

# 2. DATA IMPORT ----------------------------------------------------------------
song_pop          <- readr::read_csv("~/Documents/R/musicoset_popularity 4/song_pop.csv")
songs             <- readr::read_csv("~/Documents/R/musicoset_metadata 4/songs.csv")
acoustic_features <- readr::read_csv("~/Documents/R/musicoset_songfeatures 8/acoustic_features.csv")
song_chart        <- readr::read_csv("~/Documents/R/musicoset_popularity 4/song_chart.csv")

# 3. DATA PREPARATION -----------------------------------------------------------
musicdata <- song_pop %>%
  left_join(songs %>% select(song_id, song_name, artists), by = "song_id") %>%
  relocate(song_id, song_name, artists)

longevity <- musicdata %>%
  group_by(song_id) %>%
  summarise(years_on_chart = n_distinct(year), .groups = "drop") %>%
  mutate(long_hits = as.integer(years_on_chart > 1)) %>%
  left_join(acoustic_features, by = "song_id") %>%
  distinct(song_id, .keep_all = TRUE) %>%
  left_join(songs %>% select(song_id, song_name, artists), by = "song_id") %>%
  relocate(song_id, song_name, artists)

# Log transforms (do this ONCE)
longevity <- longevity %>%
  mutate(log_instrumentalness = log(instrumentalness + 1e-5),
         log_speechiness      = log(speechiness + 1e-5))

# 4. EXPLORATORY DATA ANALYSIS (EDA) --------------------------------------------
summary(longevity)
colSums(is.na(longevity))

# Class balance
longevity_table <- table(longevity$long_hits)
longevity_table
prop.table(longevity_table) * 100

# Consistent labels helper
hit_labels <- function(x) factor(x, levels = c(0, 1), labels = c("One-Year Hits", "Long-Term Hits"))

# Class distribution plot
ggplot(longevity, aes(x = hit_labels(long_hits), fill = hit_labels(long_hits))) +
  geom_bar(aes(y = (..count..) / sum(..count..)), width = 0.6) +
  geom_text(aes(y = (..count..) / sum(..count..),
                label = scales::percent((..count..) / sum(..count..), accuracy = 0.1)),
            stat = "count", vjust = -0.5, size = 4, fontface = "bold") +
  scale_y_continuous(labels = scales::percent_format(),
                     expand = expansion(mult = c(0, 0.1))) +
  labs(title = "Class Distribution: One-Year Hits vs Long-Term Hits",
       x = "Hit Status", y = "Percentage (%) of Songs") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_text(size = 12, margin = margin(t = 10)),
        axis.text.x  = element_text(size = 12, margin = margin(t = 9)),
        axis.title.y = element_text(size = 12, margin = margin(r = 10)),
        axis.text.y  = element_text(size = 12, margin = margin(r = 6)))

# Histogram distributions (raw)
longevity %>%
  select(danceability, energy, valence, tempo, loudness, acousticness, instrumentalness, liveness, speechiness) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~feature, scales = "free") +
  labs(title = "Histogram Distributions for Acoustic Features",
       x = "Value", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Histogram distributions 
longevity %>%
  select(log_instrumentalness, log_speechiness) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~feature, scales = "free") +
  labs(title = "Histogram Distributions (Log-Transformed Features)",
       x = "Value", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Density comparisons
longevity %>%
  pivot_longer(cols = c(acousticness, danceability, energy,
                        log_instrumentalness, liveness, loudness,
                        log_speechiness, valence, tempo),
               names_to = "feature", values_to = "value") %>%
  ggplot(aes(x = value, fill = hit_labels(long_hits))) +
  geom_density(alpha = 0.4) +
  facet_wrap(~feature, scales = "free") +
  labs(title = "Acoustic Density: One-Year Hits vs Long-Term Hits",
       x = NULL, y = NULL, fill = NULL) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.title = element_blank())

# 5. TRAIN/TEST SPLIT  --------------------------------
set.seed(123)
n <- nrow(longevity)
train_idx <- sample(seq_len(n), size = floor(0.8 * n))
train_data <- longevity[train_idx, ]
test_data  <- longevity[-train_idx, ]

# 6. LOGISTIC REGRESSION --------------------------------------------------------
long_hits_glm <- glm(long_hits ~ acousticness + danceability + energy +
                       log_instrumentalness + liveness + loudness +
                       log_speechiness + valence + tempo,
                     data = train_data, family = binomial(link = "logit"))
summary(long_hits_glm)

# VIF check
car::vif(long_hits_glm)

# Predictions + accuracy (0.5 threshold)
glm_probs <- predict(long_hits_glm, newdata = test_data, type = "response")
glm_preds <- ifelse(glm_probs > 0.5, 1, 0)
mean(glm_preds == test_data$long_hits)

# Stepwise (AIC)
step_model <- step(long_hits_glm, direction = "both", trace = 0)
summary(step_model)

step_probs <- predict(step_model, newdata = test_data, type = "response")
step_preds <- ifelse(step_probs > 0.5, 1, 0)
mean(step_preds == test_data$long_hits)

# 7. RANDOM FOREST --------------------------------------------------------------
train_rf <- train_data %>% mutate(long_hits = factor(long_hits))
test_rf  <- test_data  %>% mutate(long_hits = factor(long_hits))

rf_model <- randomForest(long_hits ~ acousticness + danceability + energy +
                           log_instrumentalness + liveness + loudness +
                           log_speechiness + valence + tempo,
                         data = train_rf, ntree = 500, importance = TRUE)
varImpPlot(rf_model, main = "Feature Importance: What Drives Longevity?")

# 8. MODEL EVALUATION -----------------------------------------------------------
rf_preds <- predict(rf_model, newdata = test_rf, type = "class")
conf_matrix <- table(Predicted = rf_preds, Actual = test_rf$long_hits)
conf_matrix

# Probabilities for class "1" (Long-Term Hits)
rf_probs <- predict(rf_model, newdata = test_rf, type = "prob")[, "1"]

# Add this to your libraries ----------------------------------------------------


# 9. AUC + ROC CURVES (Logistic Regression + Random Forest) ---------------------

# --- Logistic Regression ROC/AUC ---
# glm_probs already created above: predict(long_hits_glm, newdata = test_data, type="response")
roc_glm <- pROC::roc(response = test_data$long_hits, predictor = glm_probs, quiet = TRUE)
auc_glm <- pROC::auc(roc_glm)
auc_glm

plot(roc_glm, main = paste0("ROC Curve (Logistic Regression) | AUC = ", round(as.numeric(auc_glm), 3)))

# --- Random Forest ROC/AUC ---
# rf_probs already created above: predict(rf_model, newdata = test_rf, type="prob")[,"1"]
# test_rf$long_hits is a factor, so convert to 0/1 for ROC cleanly:
rf_actual <- as.integer(as.character(test_rf$long_hits))

roc_rf <- pROC::roc(response = rf_actual, predictor = rf_probs, quiet = TRUE)
auc_rf <- pROC::auc(roc_rf)
auc_rf

plot(roc_rf, main = paste0("ROC Curve (Random Forest) | AUC = ", round(as.numeric(auc_rf), 3)))


 
