# 🎵 Billboard Hot 100 - Acoustic Features and Longevity 🎵

## Project Overview

This project examines whether intrinsic acoustic features can explain and predict sustained chart performance on the **Billboard Hot 100**.

Using ~20,000 songs from the MusicOSet dataset (1964–2018), I model the probability that a track remains on the chart for more than one year.

The goal is not only predictive performance, but to evaluate whether measurable audio characteristics contain structurally meaningful signal related to long-term commercial success.

## Problem Definition

### Binary Classification Task

- **0**: Appeared on chart ≤ 1 year  
- **1**: Appeared on chart > 1 year  

### Class Distribution

- 77% one-year hits  
- 23% long-term hits  

This creates:
- Moderate class imbalance  
- Substantial feature overlap between classes  

## Data Sources

- Billboard chart data (MusicOSet)  
- Spotify-derived acoustic features  

**Total observations:** ~20,000 songs  
**Time span:** 1964–2018  

## Features Used

- Danceability  
- Energy  
- Loudness  
- Tempo  
- Acousticness  
- Valence  
- Instrumentalness *(log-transformed)*  
- Speechiness *(log-transformed)*  
- Liveness  

## Preprocessing

- Skewness diagnostics  
- Log transformations for heavy-tailed predictors  
- Correlation matrix inspection  
- 80/20 train-test split with fixed seed for reproducibility  

## Modelling Approach

### 1. Logistic Regression (Baseline Model)

Purpose: Establish whether average marginal effects of individual acoustic features are associated with longevity.

- Logit link function  
- AIC-based stepwise selection (`MASS::stepAIC`)  
- Coefficient interpretation  
- Out-of-sample evaluation  

### 2. Random Forest

Purpose: Capture potential threshold effects and feature interactions not explicitly modelled in the linear framework.

- 500-tree ensemble  
- Feature importance extraction  
- Confusion matrix analysis  
- ROC curve and AUC evaluation  

## Evaluation Metrics

- Accuracy  
- Confusion Matrix  
- ROC Curve  
- AUC  

## Key Findings

- Individual acoustic features exhibit substantial class overlap.  
- Logistic regression identifies statistically significant associations (e.g., danceability, loudness), but effect sizes are modest.  
- Random forest detects multivariate structure but struggles with recall for long-term hits.  
- AUC indicates moderate discriminative power, suggesting an acoustic signal is present but diffuse.  

## Interpretation

- Song longevity is not driven by a single dominant acoustic characteristic.  
- Predictive signal appears joint and overlapping rather than sharply separable.  
- Acoustic features alone are insufficient for high-confidence prediction of sustained chart success.  
- External factors (marketing, artist reputation, genre cycles, cultural timing) likely play a major role.  

## Technical Stack

- R  
- tidyverse  
- randomForest  
- MASS  
- pROC  
