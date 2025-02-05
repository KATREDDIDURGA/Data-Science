#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# In[2]:


# Read the data file from Google Colab
data = pd.read_csv('data.csv')

# Print the data
data.head()


# In[3]:


# prompt: eliminate first row to count,  because those are names, don't completely remove it

import pandas as pd
data = pd.read_csv('data.csv', skiprows=1)


# In[4]:


data.tail()


# In[5]:


# Get descriptive statistics and round the values
pande = data.describe().round(2)

# Transpose the DataFrame and display it
pande_transposed = pande.transpose()
pande_transposed


# In[6]:


data.info()

#To check the non null values count


# In[7]:


data.isnull().sum()

#To show all null variables in each variable
#if null values are available, to remove null rows
#data.dropna()
#data.fillna(data.mean()) to fill null values with mean


# In[8]:


data.duplicated().sum()
#if duplicates found, data.drop_duplicates()


# In[9]:


# Display boxplots for each column
for column in data.columns:
    plt.figure(figsize=(5, 3))
    sns.boxplot(y=data[column])
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    plt.show()


# In[10]:


# Count plot for the SEX column
plt.figure(figsize=(5, 3))
sns.countplot(x='SEX', data=data)
plt.title('Distribution of SEX')
plt.xlabel('SEX')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.show()


# In[11]:


value_counts = data['default payment next month'].value_counts(normalize=True) * 100

# Convert the series to DataFrame for easier plotting
value_counts_df = value_counts.reset_index()
value_counts_df.columns = ['Default Payment', 'Percentage']

# Plotting the bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='Default Payment', y='Percentage', data=value_counts_df)
plt.title('Percentage of Default Payment Next Month')
plt.xlabel('Default Payment Next Month')
plt.ylabel('Percentage (%)')
plt.show()


# In[12]:


# Assuming 'Education' is the column name in your dataset
education_counts = data['EDUCATION'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
education_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Education Levels')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[13]:


data = data[~data['EDUCATION'].isin([0, 5, 6])]


print("Number of rows after cleaning education column:", len(data))
print("Number of columns:", len(data.columns))


# In[14]:


data = data[data['MARRIAGE'] != 0]


# In[15]:


# Assuming 'df' is your DataFrame
print("Number of rows after cleaning marriage column:", len(data))
print("Number of columns:", len(data.columns))


# In[16]:


columns_to_check = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

for column in columns_to_check:
    data = data[data[column] != -2]



# In[17]:


# Assuming 'df' is your DataFrame
print("Number of rows after cleaning PAY columns:", len(data))
print("Number of columns:", len(data.columns))


# In[18]:


columns_to_check = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

for column in columns_to_check:
    data = data[data[column] >= 0]


# In[19]:


# Assuming 'df' is your DataFrame
print("Number of rows after cleaning BILL_AMT columns:", len(data))
print("Number of columns:", len(data.columns))


# In[20]:


# Get descriptive statistics and round the values
pande = data.describe().round(2)

# Transpose the DataFrame and display it
pande_transposed = pande.transpose()
pande_transposed


# In[21]:


# Assuming 'Education' is the column name in your dataset
education_counts = data['EDUCATION'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
education_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Education Levels')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[22]:


for column in data.columns:
    plt.figure(figsize=(5, 3))
    sns.boxplot(y=data[column])
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    plt.show()


# In[23]:


# Histogram with density plot
sns.histplot(data['LIMIT_BAL'], kde=True)
plt.title('Distribution of Credit Limits')
plt.xlabel('Credit Limit (NT dollar)')
plt.ylabel('Frequency')
plt.show()


# In[24]:


# log transformation to the LIMIT_BAL column to normalize the distribution.

data['LIMIT_BAL_log'] = np.log1p(data['LIMIT_BAL'])
sns.histplot(data['LIMIT_BAL_log'], kde=True)
plt.title('Distribution of Credit Limits (Log transformed)')
plt.xlabel('Credit Limit (NT dollar)')
plt.ylabel('Frequency')
plt.show()


# In[25]:


data.head()


# In[26]:


data['SEX'].value_counts().plot(kind = 'bar')
plt.title('Barchart of SEX')
plt.xlabel('SEX')
plt.ylabel('count')
plt.xticks(rotation=0)
plt.show()


# In[27]:


data['sex_dummy'] = data['SEX'].apply(lambda x: 1 if x == 1 else 0)
# we changed male to 1, and female to 0

# Assuming 'data' is your DataFrame
data = data.drop('SEX', axis=1)

# Now, 'SEX' has been removed from the DataFrame


# In[28]:


pande = data.describe().round(2)
pande_transposed =pande.transpose()
pande_transposed


# In[29]:


pande = data.describe().round(2)
pande_transposed =pande.transpose()
pande_transposed


# In[30]:


data['MARRIAGE'].value_counts().plot(kind ='bar')
plt.title('barchart of Marriage')
plt.xlabel('Marriage')
plt.ylabel('count')
plt.xticks(rotation=0)
plt.show()


# In[31]:


# Plotting with more bins
sns.histplot(data['AGE'], bins=30, kde=True)  # Adjust the number of bins as needed
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[32]:


MARRIAGE_DUM = pd.get_dummies(data['MARRIAGE'], prefix='MARRIAGE', drop_first=False)

# Concatenate the dummy variables back to the main dataframe
data = pd.concat([data, MARRIAGE_DUM ], axis=1)

# Optionally, you can drop the original 'Marital_status' column if it is no longer needed
data.drop('MARRIAGE', axis=1, inplace=True)

# Display the updated DataFrame to verify the changes
pande = print(data)


# In[33]:


# Print all column names to ensure the correct column name is being used
print(data.columns)


# In[34]:


data.head()


# In[35]:


payment_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
data[payment_columns] = data[payment_columns].replace(-1,0)
data.head()
#replaced -1  with 0, because -1 is paid ontime. In variable definition where 0 didn't explained. so it is better ML to understand for 0, as paid off.


# In[36]:


data = data.drop('LIMIT_BAL',axis = 1)


# In[37]:


print(data.columns)


# In[38]:


pande = data.describe().round (2)
pande_transposed = pande.transpose()
pande_transposed


# In[39]:


print(data.dtypes)


# In[40]:


pande = data.describe(include = 'all').round (2)
pande_transposed = pande.transpose()
pande_transposed


# In[41]:


data['MARRIAGE_1'] = data['MARRIAGE_1'].astype(int)
data['MARRIAGE_2'] = data['MARRIAGE_2'].astype(int)
data['MARRIAGE_3'] = data['MARRIAGE_3'].astype(int)


# In[42]:


print(data[['MARRIAGE_1', 'MARRIAGE_2', 'MARRIAGE_3']].describe())


# In[43]:


pande = data.describe(include = 'all').round (2)
pande_transposed = pande.transpose()
pande_transposed


# In[44]:


print(data.dtypes)


# In[45]:


data = data.drop(columns=['ID'])


# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns

# Compute the correlation matrix
correlation_matrix = data.corr()

# Plotting the correlation matrix
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()


# In[47]:


X = data.drop(columns=['default payment next month'])
y = data['default payment next month']


# In[48]:


X.head()


# In[49]:


y.head()


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


# In[51]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[52]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state = 42)
rf_model.fit(X_train, y_train)


# In[53]:


feature_importances = rf_model.feature_importances_


# In[54]:


# Create a DataFrame for visualization
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)


# In[55]:


# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()

# Print the feature importances
print(importance_df)


# In[56]:


y_pred = rf_model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[57]:


from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report

# Predict probabilities and evaluate with ROC and AUC
y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()




# In[58]:


# Class 0 (0.83),Class 1 (0.68): Precision for class 1 is lower than class 0, indicating that there are more false positives for class 1.
'''The models till now is indicating thepresence of false positives. i.e, eventhough the people pay(default), the model predicts as they do not pay(no default) in some cases'''


# In[59]:


# As per correlation matrix and the variable importance, we decided to remove Marriage_i variables, where i = 1,2,3; sex_dummy

X = data.drop(columns=['default payment next month','MARRIAGE_2','MARRIAGE_1','MARRIAGE_3','sex_dummy'])
y = data['default payment next month']

X.head()


# In[60]:


y.head()


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


# In[62]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[63]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state = 42)
rf_model.fit(X_train, y_train)


# In[ ]:





# In[64]:


feature_importances = rf_model.feature_importances_
# Create a DataFrame for visualization
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)
# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()

# Print the feature importances
print(importance_df)


# In[65]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Making predictions on the test set
predictions = rf_model.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))


# In[66]:


# Predict probabilities and evaluate with ROC and AUC
y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()





# In[95]:


#Deployment
get_ipython().system('pip install Flask')
import pickle


# In[96]:


# Save the trained model
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

# Save the scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)


# In[97]:





# In[98]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[67]:


#Grid search algorithm

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


# In[68]:


best_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=4,
    max_features='sqrt'  # adjusted based on warning
)
best_rf.fit(X_train, y_train)


# In[69]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Making predictions on the test set
predictions = best_rf.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))


# In[70]:


# Predict probabilities and evaluate with ROC and AUC
y_prob = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()





# In[71]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')

# Print the accuracy for each fold
print("Accuracy for each fold: ", scores)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy: ", np.mean(scores))
print("Standard deviation: ", np.std(scores))

# the output we got from 5 fold cross validation is satisfacroy and consistent.
# where we have lower standard deviation. Otherwise, we have to check the Data
# Quality and do future engineering.


# In[72]:


#!pip install xgboost
# this is another algorithm called gradient boost.


# In[73]:


from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[74]:


# Initialize the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)


# In[75]:


# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("XGBoost Model Performance")
print("Accuracy:", accuracy)
print(report)


# In[76]:


# Predict probabilities and evaluate with ROC and AUC
y_prob = xgb_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()





# In[77]:


from sklearn.metrics import precision_recall_curve, average_precision_score

# Predict probabilities and evaluate with Precision and Recall
y_prob = xgb_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
average_precision = average_precision_score(y_test, y_prob)

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (area = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()


# In[78]:


'''Trying to split original dataset into test, train, valid'''


# In[79]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into 60% training, 20% validation, and 20% test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# Here, test_size=0.5 corresponds to 0.5 * 0.4 = 0.2 of the original dataset for both validation and test sets

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Now you have X_train, X_valid, X_test, y_train, y_valid, y_test


# In[81]:


pip install imbalanced-learn


# In[82]:


# Import necessary libraries
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Print the shape of the original and SMOTE-enhanced training sets
print("Original training set shape:", X_train.shape, y_train.shape)
print("SMOTE-enhanced training set shape:", X_train_smote.shape, y_train_smote.shape)

# Print class distribution before and after SMOTE
print("\nClass distribution before SMOTE:")
print(f"Class 0: {sum(y_train == 0)}, Class 1: {sum(y_train == 1)}")
print("Class distribution after SMOTE:")
print(f"Class 0: {sum(y_train_smote == 0)}, Class 1: {sum(y_train_smote == 1)}")


# In[83]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Create and train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, rf_predictions)

# Print accuracy
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Print classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))


# In[84]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Create and train SVM model
svm_model = SVC(random_state=42)
svm_model.fit(X_train_smote, y_train_smote)

# Make predictions
svm_predictions = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, svm_predictions)

# Print accuracy
print(f"SVM Accuracy: {accuracy:.4f}")

# Print classification report
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))


# In[85]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Create Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Perform 100-fold cross-validation for Random Forest
rf_scores = cross_val_score(rf_model, X_train_smote, y_train_smote, cv=100, scoring='accuracy')

# Print results
print("Random Forest 100-Fold Cross-Validation Results:")
print(f"Mean Accuracy: {np.mean(rf_scores):.4f}")
print(f"Standard Deviation: {np.std(rf_scores):.4f}")


# In[86]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split the SMOTE-enhanced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_smote, y_train_smote, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Calculate accuracies on training and testing data
train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")


# In[87]:


import pandas as pd

# Assuming 'data' is your DataFrame and 'target' is your target variable
correlations = data.corr()
# Filter to show only correlations with the target variable
target_correlations = correlations['default payment next month'].sort_values()

# Display correlations
print(target_correlations)


# In[88]:


# import pandas as pd
# full_data = pd.concat([X, y.rename('default payment of next month')], axis=1)
# correlations = full_data.corr()

# target_correlations = correlations['default payment of next month'].sort_values()

# print("Correlations with 'Default Payment Next Month':\n", target_correlations)


# In[89]:


from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier

# Define your model
model = RandomForestClassifier()

# Setup cross-validation with shuffling
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Assuming X are features and y is the target in your data
# Check and print column names to verify the exact name of the target column
print(data.columns)


# Perform cross-validation
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print("Cross-validation scores:", scores)


# In[90]:


model.fit(X, y)
# feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
# print("Feature importances:\n", feature_importances)


# In[91]:


# Predict probabilities and evaluate with ROC and AUC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()





# In[ ]:




