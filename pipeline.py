import numpy as np
import pandas as pd
from sklearn.linear_model  import LinearRegression
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler

def run_pipeline(data_path):
    df = pd.read_csv(data_path)
    FEATURE_COLS = ["On_Target_Efficiency", "Off_Target_Count", "Total_Base_Pairs_Sequenced", "Sequencing_Depth"]
    TARGET_COL = "Precision_Score"
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.6f}")
    print(f"R-squared Score: {r2_score(y_test, y_pred):.6f}")

if __name__ == "__main__":
    print("CRISPR Pv Baseline Pipeline Ready.")
