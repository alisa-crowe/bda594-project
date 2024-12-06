import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from joblib import dump, load

# Load data
data_path = 'v11NumericIncidentPrediction.csv'
df = pd.read_csv(data_path)
df_filtered = df[df['CIBRS Offense Description'] != 'Other']

# Define features and target
X = df_filtered.drop(['CIBRS Offense Code', 'CIBRS Offense Description', 'Zip Code', 'City'], axis=1)
y = df_filtered['CIBRS Offense Description']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocessing for numerical data
numerical_features = ['Victim Age', 'Overall Race', 'Hour', 'Day of Month']
categorical_features = ['HHSA Region', 'Day of Week', 'Month']

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create an XGBoost compatible data matrix
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(objective='multi:softprob', num_class=len(label_encoder.classes_),
                                     max_depth=6, learning_rate=0.1, n_estimators=100))
])

# Fit model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

dump(model, './incident_type_model.pkl')