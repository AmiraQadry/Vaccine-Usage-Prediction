# Vaccine Usage Prediction
![](https://d2jx2rerrg6sh3.cloudfront.net/images/Article_Images/ImageForArticle_23783_16881345240577826.jpg)
This project aims to predict how likely people are to take an H1N1 flu vaccine using machine learning models. The project includes steps for exploratory data analysis (EDA), data preprocessing, and training logistic regression models using Maximum Likelihood Estimation (MLE) and Stochastic Gradient Descent (SGD).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Results](#results)

## Project Overview
Subjects receiving the same vaccine often show different levels of immune responses, and some may even present adverse side effects. Systems vaccinology can combine omics data and machine learning techniques to obtain highly predictive signatures of vaccine immunogenicity and reactogenicity. This project focuses on predicting H1N1 flu vaccine uptake.

## Dataset
The dataset includes features relevant to predicting H1N1 flu vaccine uptake, such as:
- h1n1_worry
- h1n1_awareness
- antiviral_medication
- contact_avoidance
- bought_face_mask
- wash_hands_frequently
- avoid_large_gatherings
- reduced_outside_home_cont
- avoid_touch_face
- age_bracket
- qualification
- race
- sex
- income_level
- marital_status
- housing_status
- employment
- census_msa
- no_of_adults
- no_of_children
- h1n1_vaccine (target variable)

## Installation
To run this project, you need Python 3.x and the following Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn

Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Exploratory Data Analysis
EDA helps in understanding the data distribution, relationships between variables, and handling missing values appropriately.

### Steps:
1. **Data Collection**: Load the dataset.
2. **Data Visualization**:
   - Plot distributions of key features using histograms and bar plots.
   - Visualize relationships using scatter plots.
   - Identify and handle missing values.
3. **Statistical Analysis**:
   - Calculate summary statistics for each feature (mean, median, standard deviation, etc.).

## Data Preprocessing
- Fill missing values in numerical columns with the mean.
- Fill missing values in categorical columns with the mode (most frequent value).
- Encode categorical variables using one-hot encoding.
- Scale the features using `StandardScaler`.

## Model Training and Evaluation
Train logistic regression models using Maximum Likelihood Estimation (MLE) and Stochastic Gradient Descent (SGD).

### Steps:
1. **Split Data**: Split the data into training and testing sets.
2. **Train Models**: Train logistic regression models using MLE and SGD.
3. **Evaluate Models**: Evaluate model performance using accuracy, precision, recall, F1 score, and ROC AUC.

## Results
The performance metrics for both logistic regression models (MLE and SGD) are:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| MLE   | 0.8414   | 0.7054    | 0.4301 | 0.5344   | 0.6909  |
| SGD   | 0.8414   | 0.7054    | 0.4301 | 0.5344   | 0.6909  |
