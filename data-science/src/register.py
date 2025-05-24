import argparse
import os
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

def main():
    """Main function of the script."""
    
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to model", required=True)
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    
    # Verify the model path exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model path does not exist: {args.model}")
    
    try:
        # Load the model
        print(f"Attempting to load model from: {args.model}")
        model = mlflow.sklearn.load_model(args.model)
        print("Model loaded successfully")

        # Register the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_price_regressor",
            registered_model_name="used_cars_price_prediction_model"
        )
        print("Model registered successfully")
        
    except MlflowException as e:
        print(f"MLflow error loading model: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
