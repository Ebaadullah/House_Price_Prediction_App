# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 1. Load the dataset
df = pd.read_csv("data.csv")
df.dropna(inplace=True)

# 2. Remove outliers
def remove_outliers(df, column):
    q25, q75 = np.percentile(df[column], 25), np.percentile(df[column], 75)
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]

df = remove_outliers(df, 'price')
df = remove_outliers(df, 'sqft_lot')

# 3. Feature engineering
df['year_sold'] = pd.to_datetime(df['date']).dt.year
df['house_age'] = df['year_sold'] - df['yr_built']
df['has_been_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

# 4. Drop unused columns ( floors removed)
df.drop(['date', 'yr_renovated', 'yr_built', 'street', 'country', 'floors'], axis=1, inplace=True)

# 5. Define X and y
X = df.drop("price", axis=1)
y = df["price"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Feature groups
numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                    'sqft_above', 'sqft_basement', 'house_age']
categorical_features = ['waterfront', 'view', 'condition', 'city',  'statezip','has_been_renovated']


# 8. Preprocessor and pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 9. Train and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 10. Save pipeline
with open("house_price_prediction_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model saved as house_price_prediction_model.pkl")
