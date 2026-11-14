# Bank Term Deposit Subscription Classifier
### Logistic Regression | Machine Learning Case Study

## 1. Project Overview

Can we predict whether a bank customer will subscribe to a term deposit based on their demographic details, previous interactions, and campaign information?

That was the question I explored in this case study.

The objective was to build a binary classification model using **Logistic Regression** to predict whether a customer would subscribe to a bank's term deposit (`yes` or `no`).

Rather than jumping straight into model training, I wanted to understand the dataset, examine the quality of the available information, and evaluate how well the model could distinguish between the two outcomes.

This project was completed as part of my Machine Learning coursework with **Intellipaat**.

## 2. Dataset Understanding

**Dataset:** Bank Marketing Dataset (`bank-additional-full`)

The dataset contains information collected from a bank's direct marketing campaigns.

| Attribute | Details |
|---|---|
| Initial records | 41,199 |
| Features | 20 predictors + 1 target |
| Initial columns | 21 |
| Target variable | `y` |
| Prediction task | Binary classification |
| Target classes | `yes` / `no` |

The dataset includes information such as:
- Customer demographics and employment-related details
- Previous campaign interactions
- Contact-related information
- Economic and campaign indicators
- Whether the customer subscribed to a term deposit

The target variable `y` indicates whether the customer subscribed (`yes`) or did not subscribe (`no`).

## 3. My Approach: From Raw Data to Predictions

### Step 1 — Understanding the Dataset

I started by loading the semicolon-separated CSV file and examining its structure using methods such as:

- `df.info()`
- `df.shape`
- `df.columns`
- `df.isnull().sum()`

**Why?**

Before choosing an algorithm, I wanted to understand what I was working with. Checking the dimensions, data types, and missing values helped me identify potential data quality issues before moving into preprocessing.

### Step 2 — Handling Missing Values and Duplicates

The initial inspection revealed:

- **76 missing values**
- **13 duplicate records**

I removed rows containing missing values using `dropna()` and removed duplicate records using `drop_duplicates()`.

After cleaning, the dataset contained **41,177 records and 21 columns**.

**Why?**

Missing values and duplicate records can affect the quality of the data used for model training. I wanted to establish a cleaner dataset before exploring its features and training the classifier.

### Step 3 — Exploring Numerical Features and Potential Outliers

I used boxplots to examine the numerical columns and understand their distributions.

**Why?**

I wanted to identify unusual observations and understand whether the numerical variables contained extreme values that deserved further investigation.

This step helped me look beyond basic summary statistics and visually inspect the data.

### Step 4 — Encoding Categorical Variables

Since the Logistic Regression model requires numerical input, I converted categorical columns into numerical representations using `LabelEncoder`.

**Why?**

The original dataset contained categorical information such as customer and campaign-related categories. Encoding was necessary to make these variables usable by the classification algorithm.

### Step 5 — Separating Features and Target

I separated the independent variables (`X`) from the target variable (`y`).

- `X`: Customer and campaign-related predictors
- `y`: Whether the customer subscribed to the term deposit

I then split the data into training and testing sets using an **80:20 split**, with `random_state=0`.

**Why?**

The model needs training data to learn patterns and separate testing data to evaluate how well those learned patterns perform on unseen observations.

### Step 6 — Building the Logistic Regression Classifier

I trained a Logistic Regression model using Scikit-learn.

**Why Logistic Regression?**

The objective was to predict one of two possible outcomes: subscription or non-subscription.

Logistic Regression is a supervised classification algorithm that estimates the probability of an observation belonging to a particular class. Its output can be used to classify customers into the `yes` and `no` categories.

### Step 7 — Evaluating Model Performance

After generating predictions on the test set, I evaluated the model using multiple classification metrics.

| Metric | Result |
|---|---:|
| Accuracy | 90.99% |
| Precision | 66.06% |
| Recall | 39.43% |
| F1-score | 49.39% |

I used these metrics because accuracy alone does not tell the complete story, especially when classification datasets contain unequal class distributions.

I also explored:

- **Confusion Matrix:** To understand correct and incorrect predictions for each class.
- **Classification Report:** To examine precision, recall, F1-score, and support for individual classes.

**An important observation:**

Although the model achieved approximately 91% accuracy, its recall was only 39.43%.

This means the model failed to identify a considerable proportion of actual subscribers. It made me question whether accuracy alone was giving a misleading impression of the model's performance.

## 4. An Additional Experiment: Does Duration Make a Difference?

One feature caught my attention: `duration`.

In this dataset, duration represents the length of the last contact with a customer.

I wanted to investigate how much this feature influenced the model's performance. So, I conducted an additional experiment:

1. Evaluate the original model with `duration`.
2. Train another Logistic Regression model after removing `duration`.
3. Compare accuracy, precision, recall, and F1-score.

**Why did I investigate this?**

Imagine trying to identify potential customers before a bank calls them. The duration of a call that has not happened yet cannot be known.

This raised an important machine learning question: does a feature improve model performance while also making the prediction scenario unrealistic?

The comparison helped me explore the difference between predictive performance and practical usefulness.

*The numerical results of this experiment will be documented after completing the comparison.*

## 5. What I Learned

This case study helped me understand several important aspects of classification:

- Why data inspection and cleaning should happen before model training.
- How categorical variables need to be represented for machine learning algorithms.
- Why Logistic Regression is suitable for binary classification problems.
- Why accuracy should not be used as the only evaluation metric.
- How precision and recall reveal different types of classification errors.
- Why understanding when a feature becomes available is important when designing a prediction system.
- How a model can achieve high accuracy while still missing many positive cases.

Most importantly, I learned that evaluating a model is not just about checking whether its score is high. It is about understanding what the score represents and whether the model is useful for the problem it is supposed to solve.

## 6. Limitations and Future Improvements

This implementation is an initial learning-oriented case study. There are several areas I would like to improve:

- Use `OneHotEncoder` for nominal categorical features instead of applying `LabelEncoder` to all categorical predictors.
- Build a preprocessing and modelling pipeline to make the workflow more robust.
- Apply feature scaling and investigate its effect on Logistic Regression.
- Use stratified train-test splitting to preserve class proportions.
- Investigate class imbalance and experiment with class weights and decision thresholds.
- Compare performance with and without `duration` using a consistent evaluation setup.
- Explore additional classification algorithms and compare their performance.

These improvements would help make the model more reliable and suitable for a realistic prediction scenario.

## 7. Technologies Used

- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Algorithm:** Logistic Regression
- **Environment:** Google Colab / Jupyter Notebook

## 8. Repository Structure

```text
Bank-Term-Deposit-Subscription-Classifier/
│
├── Bank_Marketing_Logistic_Regression.ipynb
└── README.md
```

## 9. Final Thoughts

I started this case study to understand how Logistic Regression works in a real classification problem. But the most useful part was not simply training the model and obtaining an accuracy score.

It was questioning the results, looking deeper into the errors, and investigating whether the features being used made sense for the actual prediction scenario.

That is what I want to carry forward into my future Machine Learning projects: not just building models, but understanding the decisions behind them.

---

*This is one of my model-wise Machine Learning case studies, documented as part of my continuous learning and practical implementation.*