import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
	# •	pandas: For handling your dataset.
	# •	OneHotEncoder: Encodes categorical variables into binary (0 or 1) columns.
	# •	LabelEncoder: Encodes labels (target variable).
	# •	ColumnTransformer: Combines different preprocessing operations (numeric vs. categorical).
	# •	Pipeline: Chains together preprocessing and modeling steps neatly.
	# •	LogisticRegression: Your classifier model.

# Example data setup
X = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Female'],
    'Married': ['No', 'Yes', 'Yes'],
    'Education': ['Graduate', 'Not Graduate', 'Graduate'],
    'Income': [50000, 40000, 30000],      # Numeric feature
    'Loan_Amount': [120000, 90000, 100000] # Numeric feature
})
y = pd.Series(['Approved', 'Declined', 'Approved'])

# Define features explicitly
categorical_features = ['Gender']
numeric_features = ['loan_amount', 'income']

# Correcting names (case-sensitive!)
categorical_features = ['Gender', 'Married', 'Education']  # adapt based on your actual dataset
numeric_features = ['income', 'loan_amount']

# Example correction (matching your example data exactly)
categorical_features = ['Gender', 'Married', 'Education']  # actual categorical fields
numeric_features = ['ApplicantIncome', 'LoanAmount']

# Label encode the target (loan approval status)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Logistic regression modeling pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Train pipeline
pipeline.fit(X, y_encoded)

# Make predictions
predictions = pipeline.predict(X)
probabilities = pipeline.predict_proba(X)[:,1]