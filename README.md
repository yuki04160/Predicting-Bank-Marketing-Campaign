# Predicting Bank Marketing Campaign
To predict marketing campaign outcome of a Portuguese banking institution, I built a Logistic Regression, KNN, Decision Tree, Random Forest, and Gradient Boosting in Python using UCI Machine Learning repository dataset [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).
## Project Background
Marketing Campaign is an important topic in every company because it can help company increase the number of clients. Since there are a great number of banks available throughout, a bank has to start campaigns to attract customers to subscribe its term deposit. 
## Exploratory Data Analysis
1. Variable identification
2. Univariate analysis
3. Bi-variate analysis
## Data Cleaning and Preprocessing
1. Outlier treatment
2. Missing value treatment
3. Delete features
4. Convert data type
5. Split training (0.7) and test (0.3)
6. Dummy & One-hot encoding
   - For logistic regression, I used dummy variables to avoid the dummy variable trap
   - For models other than logistic regression, I used one-hot encoding
7. Balance training dataset
   - SMOTE function to oversample a minority class
8. Feature scaling
   - Since KNN is a distance-based algorithm, I used StandardScaler function to rescale X
## Data Modeling
1. Logistic regression
2. K-nearest neighbors
3. Decision tree
4. Random forest (bagging)
5. Gradient boosting (boosting)</br>
Also, I used GridSearchCV function to perform cross-validation and find optimal hyperparameters.
## Classification Model Evaluation
   - Classification report (accuracy, precision, recall, f1-score)
   - ROC curve
   - Features importance
## Conclusion
Overall, I found that tree-based models performed better. Moreover, I found that even after oversampling on the minority class in the training dataset, the TPR is still not as high as the TNR. Thus, in the future, to improve the TPR of the models, I can try the following things: 
1. Deal with remaining outliers: perhaps I can rescale the data or separate outliers into another subset to build another model for those outliers.
2. Balanced training dataset using different methods: instead of oversampling the minority class, perhaps I can also try to use both oversampling and undersampling, or ADASYN (Adaptive Synthetic) algorithm to create a balanced training dataset.
