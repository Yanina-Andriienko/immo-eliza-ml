import pandas as pd
import joblib
from preprocessing import DataPreprocessor


def predict_new_data(input_data_path, model_path, output_data_path):
    # Initialize the DataPreprocessor with the path to new data
    data_preprocessor = DataPreprocessor(input_data_path)

    # Preprocess the data
    preprocessed_data = data_preprocessor.preprocess()

    # Load the trained Linear Regression model
    model = joblib.load(model_path)

    # Ensure the target variable 'price' is not in the preprocessed_data
    if 'price' in preprocessed_data.columns:
        preprocessed_data = preprocessed_data.drop('price', axis=1)

    # Make predictions on the preprocessed data
    predictions = model.predict(preprocessed_data)

    # Save the predictions to a CSV file
    predictions_df = pd.DataFrame(predictions, columns=['PredictedPrice'])
    predictions_df.to_csv(output_data_path, index=False)

    print(f"Predictions saved to {output_data_path}")


if __name__ == "__main__":
    # Path to the new data (assuming same structure as Cleaned_2.csv)
    input_data_path = 'data/Cleaned_2.csv'  # Adjust as needed

    # Path to the saved Linear Regression model
    model_path = 'linear_regression_model.joblib'  # Adjust as needed

    # Path to save the predictions
    output_data_path = 'predictions.csv'  # Adjust as needed

    # Call the prediction function
    predict_new_data(input_data_path, model_path, output_data_path)
