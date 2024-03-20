# Real Estate Price Prediction - Immo Eliza

## Project Overview

This project aims to predict real estate prices in Belgium using various machine learning models. The primary objective is to provide accurate price estimates for properties based on their features like location, area, number of bedrooms, etc.

## Dataset

The dataset used in this project contains information about real estate properties in Belgium, including details such as property type, location, living area, number of bedrooms, and more. It comprises around 30,000 houses.

## Dataset Features

| Feature              | Description                                                         |
| -------------------- | ------------------------------------------------------------------- |
| `district`           | The district where the property is located.                         |
| `price`              | The price of the property in euros.                                 |
| `state_construction` | The condition or state of the property (e.g., new, to renovate).    |
| `living_area`        | The living area of the property in square meters.                   |
| `bedrooms`           | The number of bedrooms in the property.                             |
| `bathrooms`          | The number of bathrooms in the property.                            |
| `has_garden`         | Indicates whether the property has a garden (1: Yes, 0: No).        |
| `kitchen`            | Indicates the type of kitchen in the property (1: Yes, 0: No).      |
| `fireplace`          | Indicates whether the property has a fireplace (1: Yes, 0: No).     |
| `swimmingpool`       | Indicates whether the property has a swimming pool (1: Yes, 0: No). |
| `has_terrace`        | Indicates whether the property has a terrace (1: Yes, 0: No).       |
| `has_attic`          | Indicates whether the property has an attic (1: Yes, 0: No).        |
| `has_basement`       | Indicates whether the property has a basement (1: Yes, 0: No).      |
| `epc`                | Energy performance certificate rating of the property.              |
| `area_total`         | The total area of the property in square meters.                    |

This table provides an overview of the features available in the dataset along with a brief description for each.

## Model

The project explores several machine learning models, starting with a baseline RandomForest model and experimenting with other models like Linear Regression and Polynomial regressors. The final model selection is based on performance metrics such as R² score and Mean Squared Error (MSE).

## Installation

Follow these steps to set up your project environment:

- **Clone the repository**

git clone git@github.com:Yanina-Andriienko/immo-eliza-ml.git

- **Navigate to the project directory**

cd immo-eliza-ml

- **Install the required dependencies**

pip install -r requirements.txt

Ensure you have `git` and `pip` installed on your system before running these commands.

## Usage

To train the model and make predictions, run the following scripts:

- **To preprocess the data:**

python preprocessing.py

- **To train the model:**

python train_rf.py

- **To make predictions on new data:**

python predict_rf.py -i path/to/newdata.csv -o path/to/predictions.csv

## Performance

The best-performing model achieved an R² score of 0.72 on the test set, indicating that it can explain 72% of the variance in property prices.

## Limitations

- The model relies heavily on the quality and comprehensiveness of the input data.
- It does not account for market trends or economic conditions.
- The model's predictions are specific to Belgium and may not generalize well to other regions.

## Contributors

- [Yanina Andriienko](https://www.linkedin.com/in/yanina-andriienko-7a2984287/)
