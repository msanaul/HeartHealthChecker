import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "heart.csv"
MODEL_PATH = "heart-disease-model.pkl"

df = pd.read_csv(DATA_PATH)

features = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]

X = df[features]
y = df["HeartDisease"]

categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numeric", StandardScaler(), numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        max_depth=9,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"5-fold ROC-AUC: {scores.mean():.4f}")

model.fit(X, y)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print(f"Saved model to {MODEL_PATH}")
