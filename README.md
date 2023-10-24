<img src="https://www.aihr.com/wp-content/uploads/High-employee-turnover.jpg" width="800">

# Employee_Turnover_Predicition

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project is a machine learning application that predicts employee turnover using a Random Forest model. Employee turnover, or attrition, is a significant concern for organizations. Predicting which employees are likely to leave can help companies take proactive steps to reduce turnover.

The project involves data preprocessing, model training, and prediction. It uses employee-related features like satisfaction level, last evaluation, number of projects, average monthly hours, time spent at the company, work accidents, and more to make predictions about employee turnover.

## Features

The project includes the following components:

- `app.py`: The main Python script that loads the dataset, preprocesses the data, trains a Random Forest model, and provides predictions.

- `datasets_9768_13874_HR_comma_sep.csv`: The dataset used for training and prediction. It includes employee information, such as satisfaction level, last evaluation, number of projects, average monthly hours, time spent at the company, work accidents, and more.

## Getting Started

Follow these steps to get started with the project:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/ReinaldoASilva/Employee_Turnover_Predicition.git

## Usage
To use the project, you can run the app.py script:
- python app.py
  
The script will perform the following:

Load the dataset from datasets_9768_13874_HR_comma_sep.csv.
Preprocess the data, including renaming columns and converting categorical variables to dummy variables.
Train a Random Forest model to predict employee turnover.
Provide predictions based on the model.
Dependencies
This project relies on the following Python libraries:

pandas
numpy
scikit-learn
You can install these libraries using the requirements.txt file provided.









