import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# 1. Load dataset
df = pd.read_csv("hand_data.csv", header=None)
X = df.iloc[:, :-1].values  # 63 features
y = df.iloc[:, -1].values   # Finger count labels

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Accuracy: {acc*100:.2f}%")
print("Confusion Matrix:\n", cm)

# 5. Save model
with open("finger_count_model.pkl", "wb") as f:
    pickle.dump(clf, f)
print("✅ Model saved as 'finger_count_model.pkl'")
