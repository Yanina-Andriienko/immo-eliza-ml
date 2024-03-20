import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set the pandas option
pd.set_option('future.no_silent_downcasting', True)


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = self.load_data()

    def load_data(self):
        return pd.read_csv(self.data_path)

    def remove_outliers(self, column_details):
        """
        Remove outliers for multiple columns using fixed bounds or IQR method.

        Parameters:
        column_details (dict): A dictionary with column names as keys. Values are either:
                               - Tuple of (lower_bound, upper_bound) for fixed bounds
                               - Tuple of (lower_quantile, upper_quantile, factor) for IQR method bounds
        """
        for column_name, details in column_details.items():
            if len(details) == 2:  # Fixed bounds provided
                lower_bound, upper_bound = details
            elif len(details) == 3:  # IQR method parameters provided
                lower_quantile, upper_quantile, factor = details
                Q1 = self.dataset[column_name].quantile(lower_quantile)
                Q3 = self.dataset[column_name].quantile(upper_quantile)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
            else:
                raise ValueError(
                    "Invalid details provided for column:", column_name)

            # Apply the bounds to remove outliers
            self.dataset = self.dataset[(self.dataset[column_name] >= lower_bound) &
                                        (self.dataset[column_name] <= upper_bound)]

    def encode_columns(self):
        """
        Manually encode specific columns with predefined mappings.
        """
        epc_mapping = {
            'A+_A++': 1, 'A+': 1, 'A_A+': 1, 'A++': 1, 'A': 1,
            'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'F_E': 8
        }
        state_construction_mapping = {
            'TO_RESTORE': 3, 'TO_RENOVATE': 3, 'TO_BE_DONE_UP': 3,
            'JUST_RENOVATED': 2, 'GOOD': 2, 'AS_NEW': 1
        }

        self.dataset['epc'] = self.dataset['epc'].replace(epc_mapping)
        self.dataset['state_construction'] = self.dataset['state_construction'].replace(
            state_construction_mapping)

    def drop_columns(self, columns_to_drop):
        existing_columns = [
            col for col in columns_to_drop if col in self.dataset.columns]
        self.dataset = self.dataset.drop(columns=existing_columns)
        print("Columns dropped:", existing_columns)

    def replace_nan_with_zero(self, columns_to_update):
        """
        Replace NaN values with 0 for specified columns.

        Parameters:
        columns_to_update (list): A list of column names where NaN values should be replaced with 0.
        """
        self.dataset[columns_to_update] = self.dataset[columns_to_update].fillna(
            0)
        print(self.dataset[columns_to_update].head())

    def handle_specific_nans(self):
        """
        Drop rows with NaN in specific columns and fill NaN in other columns with default values.
        """
        # Drop rows where 'bedrooms' is NaN
        self.dataset.dropna(subset=['bedrooms'], inplace=True)

        # Fill NaN values in 'bathrooms', 'epc', and 'state_construction' with default values
        self.dataset['bathrooms'] = self.dataset['bathrooms'].fillna(1)
        self.dataset['epc'] = self.dataset['epc'].fillna(-1)
        self.dataset['state_construction'] = self.dataset['state_construction'].fillna(
            -1)

    def handle_missing_values(self):
        numeric_columns = self.dataset.select_dtypes(
            include=['int64', 'float64']).columns
        self.dataset[numeric_columns] = self.imputer.fit_transform(
            self.dataset[numeric_columns])

    def filter_and_inspect(self, column_name, max_value):
        """
        Filter rows based on a maximum value condition for a specified column and inspect the value counts.

        Parameters:
        column_name (str): The name of the column to apply the filtering condition.
        max_value (int): The maximum value allowed for the specified column.
        """
        # Apply the condition to filter the DataFrame
        self.dataset = self.dataset[self.dataset[column_name] <= max_value]

        # Return or print the value counts for the specified column
        return self.dataset[column_name].value_counts()

    def scale_features(self):
        numeric_columns = self.dataset.select_dtypes(
            include=['int64', 'float64']).columns
        self.dataset[numeric_columns] = self.scaler.fit_transform(
            self.dataset[numeric_columns])

    def one_hot_encode(self, column_name):
        """
        Apply one-hot encoding to a specified categorical column and concatenate
        the result with the original DataFrame, excluding the original column.

        Parameters:
        column_name (str): The name of the categorical column to be one-hot encoded.
        """
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # Fit and transform the specified column
        column_encoded = encoder.fit_transform(self.dataset[[column_name]])
        column_encoded_df = pd.DataFrame(
            column_encoded, columns=encoder.get_feature_names_out([column_name]))

        # Ensure that the index of the encoded DataFrame matches the original DataFrame
        column_encoded_df.index = self.dataset.index

        # Drop the original categorical column from the DataFrame
        data_numeric = self.dataset.drop(column_name, axis=1)

        # Concatenate the new one-hot encoded DataFrame with the numeric DataFrame
        self.dataset = pd.concat([data_numeric, column_encoded_df], axis=1)

    def preprocess(self):

        # Drop specified columns
        columns_to_drop = ["id", "city", "postal_code",
                           "province", "subtype", "facades", 'terrace_area', 'garden_area', 'rooms', 'livingroom_surface',
                           'kitchen_surface', 'construction_year']
        self.drop_columns(columns_to_drop)

        # Define your bounds/details for each column
        column_details = {
            'price': (40000, 1000000),  # Fixed bounds
            'living_area': (0.25, 0.75, 1.5)  # IQR method parameters
        }

        # Remove outliers based on the specified details
        self.remove_outliers(column_details)
        self.encode_columns()

        # Replace NaN values with 0 in specified columns
        columns_to_update = ['has_garden', 'kitchen', 'fireplace',
                             'swimmingpool', 'has_terrace', 'has_attic', 'has_basement']
        self.replace_nan_with_zero(columns_to_update)

        # Handle NaNs in specific columns by either dropping rows or filling with default values
        self.handle_specific_nans()

        # Filter 'bathrooms' column and inspect the value counts
        bathrooms_value_counts = self.filter_and_inspect('bathrooms', 4)
        print(bathrooms_value_counts)

        # One-hot encode the 'district' column
        self.one_hot_encode('district')

        return self.dataset


data_preprocessor = DataPreprocessor('data/Cleaned_2.csv')

preprocessed_data = data_preprocessor.preprocess()
print(preprocessed_data.shape)
print(preprocessed_data.columns)
