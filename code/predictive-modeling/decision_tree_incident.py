from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Fill missing values and map categorical columns
X['Overall Race'] = X['Overall Race'].fillna("UNKNOWN").map(race_map)
X['Day of Week'] = X['Day of Week'].fillna("UNKNOWN").map(day_of_week_map)

# Encode 'HHSA Region' and 'Hour Group'
X['HHSA Region'] = label_encoder_region.fit_transform(X['HHSA Region'])
X['Hour Group'] = label_encoder_hour.fit_transform(X['Hour Group'])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Address class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a Random Forest model
clf = RandomForestClassifier(class_weight="balanced", random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))