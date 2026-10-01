from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent                      # the folder this script lives in

# Accept either file name (browsers often add " (1)" to repeat downloads)
for name in ["house_price_prediction_dataset.csv", "house_price_prediction_dataset (1).csv"]:
    if (HERE / name).exists():
        raw_path = HERE / name
        break
else:
    raise SystemExit("Raw CSV not found. Put it next to this script.")

df = pd.read_csv(raw_path)
print("Raw shape:", df.shape)

# ---- Cleaning ----
df = df.dropna(subset=["House_Price_Million_RWF"])         # no price = unusable row
df = df.drop_duplicates().reset_index(drop=True)           # remove repeated records
df.loc[df["Area_m2"] > 600, "Area_m2"] = np.nan            # impossible areas -> blank
df.loc[df["Distance_to_City_km"] > 40, "Distance_to_City_km"] = np.nan   # impossible distance
df = df[df["House_Price_Million_RWF"] <= 400].reset_index(drop=True)     # impossible prices -> drop
df.to_csv(HERE / "cleaned_house_price_dataset.csv", index=False)
print("Cleaned shape:", df.shape)

# ---- Model ----
target = "House_Price_Million_RWF"
numeric = ["Area_m2", "Bedrooms", "Bathrooms", "House_Age_Years", "Distance_to_City_km", "Parking_Spaces"]
categorical = ["Neighborhood"]
X = df[numeric + categorical]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("preprocess", ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(drop="first"))]), categorical),
    ])),
    ("regressor", LinearRegression()),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Test R2: {r2_score(y_test, pred):.4f}  MAE: {mean_absolute_error(y_test, pred):.2f}  "
      f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")

joblib.dump(model, HERE / "house_price_model.sav")
print("Saved:", HERE / "house_price_model.sav")