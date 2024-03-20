import pandas as pd
import joblib
from preprocessing import DataPreprocessor


def load_model(model_path):
    """Load the trained RandomForest model from the specified path."""
    return joblib.load(model_path)


def preprocess_new_data(data_path):
    """Preprocess new data using the DataPreprocessor class."""
    data_preprocessor = DataPreprocessor(data_path)
    preprocessed_data = data_preprocessor.preprocess()
    return preprocessed_data


def make_predictions(model, new_data):
    """Make predictions using the loaded model and preprocessed new data."""
    predictions = model.predict(new_data)
    return predictions


def save_predictions(predictions, output_path):
    """Save the predictions to a CSV file."""
    pd.DataFrame(predictions, columns=['Predicted Price']).to_csv(
        output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # Path to the Cleaned_2.csv data (to be predicted)
    new_data_path = 'data/Cleaned_2.csv'

    # Path to the saved RandomForest model
    model_path = 'random_forest_model.joblib'

    # Load the trained RandomForest model
    model = load_model(model_path)

    # Preprocess the new data
    new_data = preprocess_new_data(new_data_path).drop(
        'price', axis=1, errors='ignore')

    # Make predictions on the new data
    predictions = make_predictions(model, new_data)

    # Save or display the predictions
    save_predictions(predictions, 'predictions_on_Cleaned_2.csv')
