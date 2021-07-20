# Predicting Bank Marketing Campaign
To predict the marketing campaign outcome of a Portuguese banking institution, I built a Logistic Regression, KNN, Decision Tree, Random Forest, and Gradient Boosting in Python using UCI Machine Learning repository dataset - [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).
## Project Background
Marketing Campaign is an important topic in every company because it can help company increase the number of clients. Since there are a great number of banks available throughout, a bank has to start campaigns to attract customers to subscribe its term deposit. 
## Exploratory Data Analysis
   - Variable identification
   - Univariate analysis
   - Bi-variate analysis
## Data Cleaning and Preprocessing
   - Outlier treatment
   - Missing value treatment
   - Delete features
   - Convert data type
   - Split into training (0.7) and test (0.3)
   - Dummy & One-hot encoding
     - For logistic regression, I used dummy variables to avoid the dummy variable trap.
     - For models other than logistic regression, I used one-hot encoding.
   - Balance training dataset
     - I used SMOTE function to oversample a minority class.
   - Feature scaling
     - Since KNN is a distance-based algorithm, I used StandardScaler function to rescale X.
## Data Modeling
1. Logistic regression
2. K-nearest neighbors
3. Decision tree
4. Random forest (bagging)
5. Gradient boosting (boosting)

Also, to tune models, I used GridSearchCV function to perform cross-validation and find optimal hyperparameters.
## Classification Model Evaluation
   - Classification report (accuracy, precision, recall, f1-score)
   - ROC curve
   - Features importance
## Conclusion

Model|Accuracy|F1 score|Recall (TPR)|TNR|Precision|AUROC
-----|-----|-----|-----|-----|-----|-----
**Logistic Regression**|0.84|0.35|0.41|0.89|0.30|0.65
**KNN**|0.87|0.40|0.40|0.93|0.40|0.67
**Decision Tree**|0.89|0.44|0.41|0.95|0.48|0.68
**Random Forest**|0.89|0.41|0.35|0.96|0.49|0.65
**Gradient Boosting**|0.9|0.39|0.29|0.97|0.56|0.63

Overall, I found that tree-based models performed better. Moreover, I found that even after oversampling on the minority class in the training dataset, the TPR is still not as high as the TNR. Thus, in the future, to improve the TPR of the models, I can try the following things: 
1. Deal with the remaining outliers: since there are still some outliers that haven't been treated yet, perhaps I can rescale the data or separate those outliers into another subset to build another model.
2. Balanced training dataset using different methods: instead of oversampling the minority class, perhaps I can also try to use both oversampling and undersampling, or ADASYN (Adaptive Synthetic) algorithm to create a balanced training dataset.
