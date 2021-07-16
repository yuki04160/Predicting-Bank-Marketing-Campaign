# Predicting Bank Marketing Campaign
To predict marketing campaign outcome of a Portuguese banking institution, I built five different models in Python using UCI Machine Learning repository dataset [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).
## Introduction
Marketing Campaign is an important topic in every company because it can help company increase the number of clients. Since there are a great number of banks available throughout, a bank has to start campaigns to attract customers to subscribe its term deposit. 
## Exploratory Data Analysis
1. Variable Identification
2. Univariate Analysis
3. Bi-variate Analysis
## Data Cleaning and Preprocessing
1. Outlier Treatment
2. Missing Value Treatment
3. Delete Features
4. Convert Data Type
5. Split Training (0.7) and Test (0.3)
6. Dummy & One Hot Encoding 
7. Balance Training Dataset (oversampling)
8. Feature Scaling
## Data Modeling
1. Logistic Regression
2. Tuned KNN
3. Tuned Decision Tree
4. Tuned Random Forest (Bagging)
5. Tuned Gradient Boosting (Boosting)
## Classification Model Evaluation
   - Classification Report (accuracy, precision, recall, f1-score)
   - ROC Curve
   - Features Importance
## Conclusion
Overall, I found that tree-based models performed better. Moreover, I found that even after oversampling on the minority class in the training dataset, the TPR is still not as high as the TNR. Thus, in the future, to improve the TPR of the models, we can try the following things: 
1. Deal with remaining outliers: Perhaps I can rescale the data or separate outliers into another subset to build another model for those outliers.
2. Balanced training dataset using different methods: Instead of oversampling the minority class, perhaps I can also try to use both oversampling and undersampling, or ADASYN (Adaptive Synthetic) algorithm to create a balanced training dataset.
