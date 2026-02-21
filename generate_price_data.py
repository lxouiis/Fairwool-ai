import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# --- synthetic generation ---
rng = np.random.default_rng(42)
n = 600
grade = rng.integers(1, 4, size=n) # 1=C, 2=B, 3=A
weight_kg = rng.uniform(0.5, 5.0, size=n)
season_idx = rng.choice([-0.05, 0.0, 0.10], size=n) # monsoon, normal, winter
distance_km = rng.uniform(0, 80, size=n)
contamination = rng.uniform(0.0, 0.9, size=n)

base = np.where(grade==3, 55, np.where(grade==2, 35, 20))
price = base * (1 + season_idx) * (1 - 0.35*contamination) - 0.10*distance_km + rng.normal(0, 2.0, size=n)
price = np.clip(price, 5, None) # prevent negative

df = pd.DataFrame({
    "grade": grade,
    "weight_kg": weight_kg,
    "season_idx": season_idx,
    "distance_km": distance_km,
    "contamination": contamination,
    "price_per_kg": price
})

print("Generated synthetic data (first 5 rows):")
print(df.head())

# --- train/val split ---
train = df.sample(frac=0.8, random_state=42)
val = df.drop(train.index)

X_train = train[["grade","weight_kg","season_idx","distance_km","contamination"]].to_numpy()
y_train = train["price_per_kg"].to_numpy()

X_val = val[["grade","weight_kg","season_idx","distance_km","contamination"]].to_numpy()
y_val = val["price_per_kg"].to_numpy()

reg = LinearRegression()
reg.fit(X_train, y_train)

pred = reg.predict(X_val)
mae = mean_absolute_error(y_val, pred)
print("MAE:", mae)

# Save coefficients for explainability
print("coef:", reg.coef_, "intercept:", reg.intercept_)

# Save the model
joblib.dump(reg, "price_reg.joblib")
print("Saved price model to price_reg.joblib")
