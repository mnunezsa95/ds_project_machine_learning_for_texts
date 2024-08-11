# The Film Junky Union - Movie Review Classifier (Machine Learning for Texts)

## Overview
The Film Junky Union is an edgy and innovative community for classic movie enthusiasts. This project focuses on building a system to filter and categorize movie reviews, helping the community better understand and manage the sentiment of the content shared by its members.

## Description
This project aims to develop a machine learning model that can automatically identify and classify movie reviews as positive or negative. By leveraging a dataset of IMDB movie reviews with polarity labels, the model is trained to recognize sentiment patterns and classify reviews accordingly. The ultimate goal is to create a reliable and efficient tool that meets the community's needs, ensuring that negative reviews are accurately identified.

## Objectives
- Develop a Classification Model: Build a model that can accurately classify movie reviews as either positive or negative.
- Achieve a High F1 Score: The model must achieve an F1 score of at least 0.85, ensuring a balance between precision and recall in the classification.
- Enhance Community Experience: Provide The Film Junky Union with a tool that enhances the way reviews are filtered and categorized, contributing to a more organized and enjoyable community experience.

## Libraries
The following libraries and tools were used in the development of this movie review classifier:

- os, math, random, re: Core Python libraries used for general-purpose functions like file handling, mathematical operations, random number generation, and regular expression processing.
- spacy: An advanced natural language processing (NLP) library used for tokenization, lemmatization, and other text preprocessing tasks.
- nltk: The Natural Language Toolkit, used for text processing tasks such as tokenization, stopword removal, and lemmatization.
- torch: A deep learning framework used for building and training neural networks, especially useful for working with large datasets and complex models.
- transformers: A library by Hugging Face that provides pre-trained models and tools for working with transformer-based models like BERT, used in advanced NLP tasks.
- NumPy: A fundamental package for numerical computing in Python, used for efficient array manipulation and mathematical operations.
- Pandas: A powerful data manipulation and analysis library, used for handling the IMDB dataset, including data cleaning and preprocessing.
- matplotlib: Visualization libraries used for plotting graphs and visualizing trends in the data, including dates.
- seaborn: A statistical data visualization library built on top of matplotlib, used for creating visually appealing and informative plots.
- sklearn (Scikit-learn): A machine learning library providing tools for model training, evaluation, and selection, including classifiers, feature extraction, and cross-validation.
  - DummyClassifier: Used as a baseline model for comparison.
  - LogisticRegression: A simple yet effective model used in text classification tasks.
  - TfidfVectorizer: For transforming text data into TF-IDF feature vectors.
- LightGBM (LGBMClassifier): A gradient boosting framework used for building efficient and high-performing models, especially in classification tasks.

## Conclusion
The Film Junky Union's initiative to develop a system for filtering and categorizing movie reviews through automated detection of negative reviews has yielded promising results. Among the models evaluated using the IMDB movie reviews dataset, the logistic regression models demonstrated strong and consistent performance across various metrics, including accuracy, F1 score, Area Under the Precision-Recall Curve (APS), and Receiver Operating Characteristic Curve (ROC AUC).

Key insights from the evaluations include:
- Model 1: Displayed high accuracy and strong performance metrics, but a slight drop in the test set performance suggested potential overfitting.
- Model 3: Achieved an accuracy of 0.93 on the training set and 0.88 on the test set, with F1 scores of 0.93 and 0.88, respectively. APS and ROC AUC values were notably high, indicating the model's robust discrimination capability.
- Model 4: Showed balanced performance with an accuracy of 0.93 on the training set and 0.85 on the test set. F1 scores were consistent at 0.93 and 0.86, and both APS and ROC AUC values were high, confirming the model's strong generalization to unseen data.
- Model 9: Due to technical limitations on the user's M1 MacBook Pro, the analysis of the BERT model could not be completed.

When using the model to predict new reviews, all three logistic regression models were relatively successful at identifying reviews as either negative (0) or positive (1). They were able to tag new reviews appropriately, demonstrating their practical applicability and reliability.

Overall, the logistic regression models proved effective in identifying patterns and making accurate predictions, meeting the project's goal with an F1 score of at least 0.85. Despite the limitations encountered with the BERT model, the logistic regression models provide a reliable foundation for The Film Junky Union's system, effectively categorizing movie reviews with high precision and recall.
