# Real Estate Price Prediction - Immo Eliza

## Project Overview

This project aims to predict real estate prices in Belgium using various machine learning models. The primary objective is to provide accurate price estimates for properties based on their features like location, area, number of bedrooms, etc.

## Dataset

The dataset used in this project contains information about real estate properties in Belgium, including details such as property type, location, living area, number of bedrooms, and more. It comprises around 30,000 houses.

## Features

- `location`: Location of the property
- `living_area`: Living area in square meters
- (Include other features used in your model)

## Model

The project explores several machine learning models, starting with a baseline Linear Regression model and experimenting with more complex models like RandomForest and Polynomial regressors. The final model selection is based on performance metrics such as R² score and Mean Squared Error (MSE).

## Installation

To set up the project environment:

1. Clone the repository

git clone git@github.com:Yanina-Andriienko/immo-eliza-ml.git

2. Navigate to the project directory

cd immo-eliza-ml

3. Install the required dependencies

pip install -r requirements.txt

## Usage

To train the model and make predictions, run the following scripts:

- To preprocess the data:

python preprocessing.py

- To train the model:

python train.py

- To make predictions on new data:

python predict.py -i path/to/newdata.csv -o path/to/predictions.csv

## Performance

The best-performing model achieved an R² score of 0.57 on the test set, indicating that it can explain 57% of the variance in property prices.

## Limitations

- The model relies heavily on the quality and comprehensiveness of the input data.
- It does not account for market trends or economic conditions.
- The model's predictions are specific to Belgium and may not generalize well to other regions.

## Contributors

- [Yanina Andriienko](https://www.linkedin.com/in/yanina-andriienko-7a2984287/)
