import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load data
data_path = 'v11NumericIncidentPrediction.csv'
df = pd.read_csv(data_path)
df_filtered = df[df['CIBRS Offense Description'] != 'Other']

# Define features and target
X = df_filtered.drop(['CIBRS Offense Code', 'CIBRS Offense Description', 'Zip Code', 'City'], axis=1)
y = df_filtered['CIBRS Offense Description']

# Preprocessing for numerical data
numerical_features = ['Victim Age', 'Overall Race', 'Hour', 'Day of Month']
categorical_features = ['Agency', 'CIBRS Status', 'HHSA Region', 'Day of Week', 'Month']

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost compatible data matrix
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', xgb.XGBClassifier(objective='multi:softprob', num_class=y.nunique(), use_label_encoder=False))])

# Fit model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
feature_importances = model.named_steps['classifier'].feature_importances_
# Get feature names from the preprocessor
encoded_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(input_features=categorical_features)
all_features = numerical_features + list(encoded_features)

# Plot feature importances
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.barh(all_features, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importances for XGBoost Model')
plt.show()
