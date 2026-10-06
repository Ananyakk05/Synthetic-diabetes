# Synthetic Diabetes Data Generation Using GenAI

## Overview

This project focuses on generating synthetic diabetes data using generative AI and deep learning techniques. The aim is to create realistic synthetic datasets that can be used for analysis and experimentation while preserving the statistical characteristics of the original data.

## Technologies Used

- Python
- NumPy
- Pandas
- PyTorch
- Scikit-learn
- Matplotlib
- Joblib
- GAN
- VAE

## Project Features

- Preprocessing of diabetes data
- Generation of synthetic data using GAN and VAE approaches
- Comparison of real and synthetic data
- Visualization of data distributions
- Saving generated datasets as CSV files

## Dataset

The project works with diabetes-related data and generates synthetic records based on the patterns learned from the original dataset.

## Output

The project generates synthetic datasets using different approaches:

- `synthetic_diabetes_gan.csv`
- `synthetic_diabetes_vae.csv`

A comparison visualization is provided in:

- `distribution_comparison.png`

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt