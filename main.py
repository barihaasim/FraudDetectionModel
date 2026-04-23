#importing libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

#loading data
data = pd.read_csv("creditcardinfo.csv")

print("First 5 rows:")
print(data.head())

print("\nClass distribution:")
print(data['is_fraud'].value_counts())


plt.figure(figsize=(10, 6))
ax = sns.countplot(x='is_fraud', data=data, hue='is_fraud', palette=['green', 'red'], legend=False)
plt.title("Fraud vs Normal Transactions", fontsize=14, fontweight='bold')
plt.xlabel("0 = Normal, 1 = Fraud", fontsize=12)
plt.ylabel("Count", fontsize=12)

#count
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height,
            f'{int(height)}',
            ha="center", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
plt.close() 



X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# converting categorical data into numberica
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)


# handling imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", X_train.shape)


#traininng model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#making predictions
y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nSample Predictions:")
print(results.head(10))



cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.close()  


# calcluating the probability of fraud for ROC curve
y_pred_proba = model.predict_proba(X_test)[:, 1] 
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.show()
plt.close() 


#saving model
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "fraud_model.pkl")

joblib.dump(model, save_path)
print(f"File location: {save_path}")

