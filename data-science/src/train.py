# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""
import os
import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth of the trees")
    return parser.parse_args()

def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder"""
    files = os.listdir(path)
    assert len(files) == 1, f"Expected 1 file in {path}, found {len(files)}"
    return os.path.join(path, files[0])

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''
    # Load data
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Validate target column
    if "Price" not in train_df.columns:
        raise ValueError("Target column 'Price' not found in training data")

    # Split features/target
    y_train = train_df["Price"]
    X_train = train_df.drop(columns=["Price"])
    y_test = test_df["Price"]
    X_test = test_df.drop(columns=["Price"])

    # Train model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Log parameters and metrics
    mlflow.log_params({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth
    })
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("MSE", mse)

    # Save model
    os.makedirs(args.model_output, exist_ok=True)
    mlflow.sklearn.save_model(model, args.model_output)

if __name__ == "__main__":
    with mlflow.start_run():
        args = parse_args()
        print(f"Train data path: {args.train_data}")
        print(f"Test data path: {args.test_data}")
        print(f"Model output path: {args.model_output}")
        print(f"n_estimators: {args.n_estimators}, max_depth: {args.max_depth}")
        main(args)
