# Insurance Charge Prediction Using Linear Regression

## Project Overview

I worked on an insurance charge prediction problem to understand how Linear Regression could be used to predict insurance charges based on different customer-related features.

Instead of directly training a model, I first explored the dataset, checked its quality, investigated potential issues, and then prepared the data for modelling.

## My Approach

### 1. Understanding the Data

I started by checking the dataset's shape, column names, data types, missing values, and duplicate records.

I found missing values in the dataset, so I handled them before proceeding. I used mean imputation for numerical columns and mode imputation for categorical columns.

### 2. Exploring the Data and Outliers

After cleaning the missing values, I explored the data using visualizations to understand the distributions of different features.

I used boxplots and the IQR method to investigate potential outliers. This helped me identify unusual values and examine whether the data needed further treatment.

### 3. Checking Multicollinearity

Next, I calculated the Variance Inflation Factor (VIF) to investigate whether some independent variables were highly correlated with one another.

Based on this analysis, I explored removing selected features to understand how multicollinearity and feature selection could affect the model.

### 4. Preparing Categorical Features

The dataset contained categorical features such as sex, smoking status, and region. Since these features were not all numerical, I explored categorical encoding to prepare them for model training.

### 5. Training the Model

Once I had prepared the features, I split the data into training and testing sets using an 80:20 ratio and `random_state=7`.

I then trained a Linear Regression model to predict insurance charges using the selected features.

### 6. Evaluating the Results

I evaluated the trained model using the R² score and obtained approximately **0.86** in my recorded experiment.

This gave me an initial understanding of how well the model explained the variation in insurance charges on the evaluation data. I also learned that R² alone does not tell the complete story of prediction performance.

## What I Learned

- Why checking missing values is important before training a model.
- How boxplots and the IQR method can help investigate outliers.
- How VIF helps identify potential multicollinearity.
- Why categorical features need appropriate encoding.
- How to split data into training and testing sets.
- How to train a Linear Regression model and interpret its R² score.

## Future Improvements

I would like to revisit this case study to compare different preprocessing approaches, evaluate the model using MAE and RMSE, and build a reusable prediction workflow for new customer inputs.

## Acknowledgment

This case study was completed as part of my Machine Learning learning journey through coursework and hands-on implementation.