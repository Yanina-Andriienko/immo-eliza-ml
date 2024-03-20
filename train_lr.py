from preprocessing import DataPreprocessor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
import joblib 


def train_linear_regression(preprocessed_data):
    """
    Trains a LinearRegression model on the preprocessed dataset.

    Parameters:
    preprocessed_data (DataFrame): A DataFrame containing the preprocessed data.
    """

    # Split the data into features and target variable
    X = preprocessed_data.drop('price', axis=1)  # Features
    y = preprocessed_data['price']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initialize the Linear Regression model
    linear_regression = LinearRegression()

    # Fit the model to the training data
    linear_regression.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = linear_regression.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Save the model
    joblib.dump(linear_regression, 'linear_regression_model.joblib')


if __name__ == "__main__":
    # Initialize the DataPreprocessor
    data_preprocessor = DataPreprocessor('data/Cleaned_2.csv')

    # Preprocess the data
    preprocessed_data = data_preprocessor.preprocess()

    # Use the preprocessed_data to train model
    train_linear_regression(preprocessed_data)
