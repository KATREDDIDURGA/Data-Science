# Wildfire Hotspot Prediction using Advanced Deep Learning Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of advanced deep learning models for wildfire hotspot prediction using NASA MODIS satellite data (2000-2023). The project compares LSTM, LSTM with Attention, and Time Series Transformer architectures for temporal pattern recognition in wildfire data.

## Table of Contents
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [License](#license)

## Key Features
- **Three Model Architectures**:
  - Baseline LSTM
  - LSTM with Attention Mechanism
  - Time Series Transformer
- **Temporal Pattern Analysis** of 23 years of MODIS data
- **Hyperparameter Optimization** using Optuna
- **Comparative Performance Evaluation** (2010-2023)

## Dataset
**NASA MODIS Fire Data** (2000-2023) containing:
- 1.4M+ wildfire records
- Key Features:
  - Latitude/Longitude
  - Brightness temperature
  - Fire Radiative Power (FRP)
  - Confidence levels
  - Day/Night indicators

Preprocessing steps include temporal aggregation, MinMax scaling, and train-test split (2010-2020 training, 2021-2023 testing).

## Methodology

### Model Architectures
1. **Baseline LSTM**:
   - Stacked LSTM layers
   - Dropout regularization
   - Optimized with Optuna

2. **LSTM with Attention**:
   - Attention mechanism for temporal weighting
   - Enhanced feature extraction

3. **Time Series Transformer**:
   - Multi-head self-attention
   - Encoder-decoder architecture
   - Positional encoding

### Evaluation Metrics
- Root Mean Squared Error (RMSE)
- Training/Validation Loss Analysis

## Results

### Performance Comparison (RMSE)
| Model                 | 2021   | 2022   | 2023   |
|-----------------------|--------|--------|--------|
| Baseline LSTM         | 55.64  | 43.63  | 141.57 |
| LSTM with Attention   | 54.26  | 42.78  | 147.89 |
| **Time Series Transformer** | **49.18** | **34.78** | **144.25** |

### Key Findings
- Transformers reduced RMSE by up to 20.29% compared to baseline LSTM
- Attention mechanisms improved temporal pattern recognition
- Models captured seasonal wildfire trends effectively

![Prediction Visualization](images/prediction_example.png) *Example prediction vs actual comparison*

## Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/wildfire-prediction.git
cd wildfire-prediction
