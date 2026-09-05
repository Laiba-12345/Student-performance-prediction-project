"""
Student Performance Prediction
A beginner-friendly Machine Learning project using scikit-learn.

Goal:
Predict a student's final exam score from study habits and academic factors.

Model:
Random Forest Regressor

Dataset:
A synthetic but realistic dataset is generated locally so the project
runs without downloading a dataset from the internet.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def create_dataset(n_students=500):
    """Create a reproducible sample student dataset."""
    study_hours = np.random.uniform(1, 10, n_students)
    attendance = np.random.uniform(50, 100, n_students)
    previous_score = np.random.uniform(35, 95, n_students)
    assignments_completed = np.random.randint(2, 11, n_students)
    sleep_hours = np.random.uniform(4.5, 9, n_students)

    noise = np.random.normal(0, 4, n_students)

    final_score = (
        0.55 * previous_score
        + 2.6 * study_hours
        + 0.20 * attendance
        + 1.3 * assignments_completed
        + 1.5 * sleep_hours
        - 25
        + noise
    )

    final_score = np.clip(final_score, 0, 100)

    return pd.DataFrame({
        "study_hours": study_hours.round(2),
        "attendance": attendance.round(2),
        "previous_score": previous_score.round(2),
        "assignments_completed": assignments_completed,
        "sleep_hours": sleep_hours.round(2),
        "final_score": final_score.round(2),
    })


def main():
    # 1. Load/create data
    df = create_dataset()
    df.to_csv("student_performance_data.csv", index=False)

    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    # 2. Separate features and target
    X = df.drop("final_score", axis=1)
    y = df["final_score"]

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # 4. Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        max_depth=10
    )
    model.fit(X_train, y_train)

    # 5. Predict
    y_pred = model.predict(X_test)

    # 6. Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance")
    print("-----------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}")

    # 7. Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nFeature Importance:")
    print(importance)

    plt.figure(figsize=(8, 5))
    importance.sort_values().plot(kind="barh")
    plt.title("Feature Importance - Student Performance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()

    # 8. Example prediction
    new_student = pd.DataFrame([{
        "study_hours": 7,
        "attendance": 90,
        "previous_score": 78,
        "assignments_completed": 9,
        "sleep_hours": 7
    }])

    predicted_score = model.predict(new_student)[0]

    print("\nExample Student")
    print("---------------")
    print(new_student.to_string(index=False))
    print(f"\nPredicted final score: {predicted_score:.2f}/100")


if __name__ == "__main__":
    main()
