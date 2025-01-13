# Sarcasm Detection Model using NLP

This project is a self-learning exercise aimed at building a sarcasm detection model using Natural Language Processing (NLP) techniques. The goal is to process textual headlines from a dataset and predict whether the headline is sarcastic or not. The project applies NLP preprocessing, exploratory data analysis (EDA), and machine learning algorithms to create and evaluate the model's performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Results](#results)
- [Workflow Explanation](#workflow-explanation)  {{ edit_1 }}
- [Understanding Sarcasm Detection](#understanding-sarcasm-detection)  {{ edit_2 }}

## Workflow Explanation
The workflow of the sarcasm detection model consists of several sequential steps:

1. **Data Loading**: The dataset, which is a JSON file containing headlines and their corresponding sarcasm labels, is loaded into a Pandas DataFrame.

2. **Data Preprocessing**: The text data undergoes preprocessing, which includes:
   - **URL Removal**: Any URLs present in the headlines are removed.
   - **Special Character Removal**: Non-alphabetical characters are stripped from the text.
   - **Lowercasing**: All text is converted to lowercase to ensure uniformity.
   - **Whitespace Stripping**: Extra whitespaces are eliminated.

3. **Exploratory Data Analysis (EDA)**: The data is analyzed to understand its structure and distribution. This includes:
   - Visualizing the class distribution of sarcastic and non-sarcastic headlines.
   - Analyzing headline length statistics.
   - Generating word clouds and extracting top N-grams.

4. **Model Training**: The model is trained using the following steps:
   - **TF-IDF Vectorization**: The headlines are vectorized using `TfidfVectorizer` to convert text into numerical features.
   - **Random Forest Classifier**: A Random Forest classifier is employed to predict sarcasm based on the extracted features.
   - **Class Balancing**: The dataset is balanced using `RandomOverSampler` to address any class imbalance.
   - **Hyperparameter Tuning**: `GridSearchCV` is utilized to optimize the hyperparameters of the Random Forest model.

5. **Model Evaluation**: The model's performance is evaluated on both training and test datasets using various metrics, including accuracy, precision, recall, and F1 score. The best model is saved for future use.

6. **Usage**: To train the sarcasm detection model, ensure that the dataset is located in the same directory as the script, and run the model training script.

## Understanding Sarcasm Detection
Sarcasm detection involves identifying statements that convey a meaning opposite to their literal interpretation. This is a complex task because sarcasm often relies on context, tone, and cultural nuances that are not explicitly stated in the text.

### Real-Time Problem Example
Consider the following scenario:

**Context**: A person is discussing the weather with a friend. The day has been gloomy, and it has been raining heavily.

**Example Statement**: "Oh great! Another rainy day!"

#### Breakdown of the Example:
1. **Literal Interpretation**: At face value, the statement seems to express excitement about the rain. However, the context suggests otherwise.
  
2. **Contextual Clue**: The phrase "Oh great!" is often used sarcastically, especially when paired with a negative situation like continuous rain. The speaker's tone (if spoken) would likely convey frustration or disappointment rather than joy.

3. **Detection Mechanism**:
   - **Word Choice**: The model analyzes the words "Oh great!" in conjunction with the context of rain. It recognizes that this phrase is commonly used in sarcastic remarks.
   - **Sentiment Analysis**: The model may also employ sentiment analysis to gauge the overall sentiment of the statement. In this case, the sentiment associated with "rainy day" is negative, which contrasts with the positive sentiment implied by "Oh great!".
   - **Training Data**: The model learns from a dataset of labeled examples where similar phrases are marked as sarcastic or non-sarcastic. This helps it identify patterns and make predictions on new, unseen data.

4. **Challenges**: 
   - **Ambiguity**: Not all statements that use similar phrases are sarcastic. The model must differentiate between genuine excitement and sarcasm, which can be challenging without additional context.
   - **Cultural Differences**: Sarcasm can vary significantly across cultures and languages. A phrase that is sarcastic in one culture may not be perceived the same way in another.

In summary, sarcasm detection is a nuanced task that requires understanding not just the words used, but also the context, tone, and cultural implications behind them. The model aims to mimic this understanding by learning from a diverse set of examples.

## Project Overview
This project uses a sarcastic headlines dataset to train a model that detects sarcasm in text. It covers a variety of NLP techniques, including text preprocessing, feature extraction with TF-IDF, and model evaluation. The model is built using a Random Forest classifier and further enhanced with hyperparameter tuning and class balancing techniques like oversampling.

## Installation
To run this project, you need to install the following dependencies:

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn nltk wordcloud joblib

Additionally, you will need the dataset, which is a JSON file containing headlines and their corresponding sarcasm labels.

## Dataset
The dataset used for this project is a JSON file named `Sarcasm_Headlines_Dataset.json`. Each entry contains a headline and a label indicating whether it's sarcastic (1) or non-sarcastic (0).

**Sample format of the dataset:**

{"headline": "This is the best day of my life", "is_sarcastic": 1}

## Preprocessing
The preprocessing steps applied to the text data are as follows:
- **URL Removal:** URLs in the headlines are removed.
- **Special Character Removal:** Non-alphabetical characters are removed.
- **Lowercasing:** All text is converted to lowercase.
- **Whitespace Stripping:** Extra whitespaces are removed.

## Exploratory Data Analysis (EDA)
The EDA steps include:
- **Class Distribution:** Visualizing the distribution of sarcastic and non-sarcastic headlines.
- **Headline Length Statistics:** Analyzing the distribution of headline lengths.
- **Word Clouds:** Visualizing the most frequent words in sarcastic and non-sarcastic headlines.
- **Top N-grams:** Extracting and displaying the top bigrams and trigrams from the headlines.

## Model Training
The model is built using the following steps:
- **TF-IDF Vectorization:** The headlines are vectorized using `TfidfVectorizer` to represent the text as numerical features.
- **Random Forest Classifier:** A Random Forest classifier is used to predict sarcasm based on the features extracted from the headlines.
- **RandomOverSampler:** The dataset is balanced using `RandomOverSampler` to address class imbalance.
- **Hyperparameter Tuning:** `GridSearchCV` is used to tune hyperparameters, optimizing the Random Forest model.

## Model Evaluation
The model is evaluated on both training and test datasets using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report

The best model is saved as `sarcasm_model.pkl` for later use.

## Usage
To train the sarcasm detection model, ensure that the dataset is located in the same directory as the script, and run the following command:

python model.ipynb

This will:
- Load the dataset.
- Perform EDA.
- Preprocess the headlines.
- Train the model using the Random Forest classifier.
- Evaluate the model on the test and training data.
- Save the trained model to disk.

## Results
After running the model, the following evaluation metrics are produced:

### Training Data Metrics:
- **Accuracy:** 0.86
- **Precision:** 0.98
- **Recall:** 0.70
- **F1 Score:** 0.81

**Classification Report (Training Data):**

              precision    recall  f1-score   support

           0       0.81      0.99      0.89     10489
           1       0.98      0.70      0.81      8207

    accuracy                           0.86     18696
   macro avg       0.89      0.84      0.85     18696
weighted avg       0.88      0.86      0.86     18696

### Test Data Metrics:
- **Accuracy:** 0.73
- **Precision:** 0.77
- **Recall:** 0.55
- **F1 Score:** 0.64

**Classification Report (Test Data):**

              precision    recall  f1-score   support

           0       0.71      0.87      0.78      4496
           1       0.77      0.55      0.64      3517

    accuracy                           0.73      8013
   macro avg       0.74      0.71      0.71      8013
weighted avg       0.74      0.73      0.72      8013