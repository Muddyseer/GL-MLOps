import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("Reading data")
    df = pd.read_csv(args.input_data)

    # Convert Segment to numerical (0 for non-luxury, 1 for luxury)
    df['Segment'] = df['Segment'].apply(lambda x: 1 if 'luxury' in x.lower() else 0)

    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_train_ratio,
        random_state=42
    )

    # Output paths are mounted as folder, hence we add filename to the path
    train_df.to_csv(os.path.join(args.train_data, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test_data.csv"), index=False)

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
