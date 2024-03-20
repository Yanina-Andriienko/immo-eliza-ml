from preprocessing import DataPreprocessor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # For saving the model


def train_random_forest(preprocessed_data):
    """
    Trains a RandomForestRegressor on the preprocessed dataset.

    Parameters:
    preprocessed_data (DataFrame): A DataFrame containing the preprocessed data.
    """

    # Split the data into features and target variable
    X = preprocessed_data.drop('price', axis=1)  # Features
    y = preprocessed_data['price']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model to the training data
    random_forest.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = random_forest.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Feature Importance
    feature_importances = pd.DataFrame(random_forest.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    print(feature_importances)

    # Save the model
    joblib.dump(random_forest, 'random_forest_model.joblib')


if __name__ == "__main__":
    # Initialize the DataPreprocessor
    data_preprocessor = DataPreprocessor('data/Cleaned_2.csv')

    # Preprocess the data
    preprocessed_data = data_preprocessor.preprocess()

    # Now, use the preprocessed_data to train your model
    train_random_forest(preprocessed_data)
