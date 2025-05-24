import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import joblib

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--max_depth", required=False, default=None, type=int)
    parser.add_argument("--model_output", type=str, help="path to model output")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("Loading data")
    train_df = pd.read_csv(os.path.join(args.train_data, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(args.test_data, "test_data.csv"))

    # Separate features and target
    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    print("Training model")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Evaluating model")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)

    # Save model in MLflow format
    mlflow.sklearn.save_model(model, "model_output")

    # Register the model with MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_price_regressor",
        registered_model_name="used_cars_price_prediction_model"
    )

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()

