# Import Libraries
import pandas as pd
from imblearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np

# Read and preprocess data
data_file_path = r'C:\Users\abdog\DEDA_Project\FraudDetection-master\data_FraudDetection_JAR2020.csv'
data = pd.read_csv(data_file_path)

train_data = data[(data['fyear'] >= 1991) & (data['fyear'] <= 1999)]
valid_data = data[(data['fyear'] >= 2000) & (data['fyear'] <= 2001)]

X_train = train_data.iloc[:, 4:]
y_train = train_data['misstate']
X_valid = valid_data.iloc[:, 4:]
y_valid = valid_data['misstate']

# Handle serial frauds
valid_paaers = valid_data[valid_data['misstate'] != 0]['p_aaer'].unique()
train_data.loc[train_data['p_aaer'].isin(valid_paaers), 'misstate'] = 0
X_train = train_data.iloc[:, 4:]
y_train = train_data['misstate']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)

# Train the model
rusboost = RUSBoostClassifier(
    estimator=DecisionTreeClassifier(min_samples_leaf=5),
    n_estimators=300,
    learning_rate=0.1,
    random_state=0
)
rusboost.fit(X_train, y_train)

# Predict and evaluate
y_pred = rusboost.predict(X_valid)
y_dec_values = rusboost.predict_proba(X_valid)[:, 1]
auc_score = roc_auc_score(y_valid, y_dec_values)
print(f'AUC: {auc_score:.4f}')

# Visualization functions
def plot_roc_curve(y_true, y_dec_values):
    fpr, tpr, _ = roc_curve(y_true, y_dec_values)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def plot_precision_recall_curve(y_true, y_dec_values):
    precision, recall, _ = precision_recall_curve(y_true, y_dec_values)
    avg_precision = average_precision_score(y_true, y_dec_values)
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.title('Feature Importance')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# Visualize results
plot_roc_curve(y_valid, y_dec_values)
plot_precision_recall_curve(y_valid, y_dec_values)
plot_feature_importance(rusboost, train_data.columns[4:])
plot_confusion_matrix(y_valid, y_pred)
