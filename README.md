# Sonar Rock vs Mine Prediction using Multi-Model Approach

## Overview

This project implements a multi-model approach to predict whether an object is a rock or a mine based on sonar data. It utilizes various machine learning algorithms to classify sonar signals, comparing their performance to identify the most effective model for this task.

## Dataset

The dataset used in this project is the Sonar dataset, which contains patterns obtained by bouncing sonar signals off a metal cylinder (mine) and a roughly cylindrical rock at various angles and under various conditions.

- Number of Instances: 208
- Number of Attributes: 60
- Target Classes: Rock (R) and Mine (M)

## Project Structure

The project follows this workflow:

1. Data Loading and Preprocessing
2. Exploratory Data Analysis
3. Feature Engineering
4. Model Training and Evaluation
5. Hyperparameter Tuning
6. Model Comparison
7. Best Model Selection and Final Prediction

## Models Implemented

The following models are implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine (SVM)
- Neural Network (MLPClassifier)
- Gradient Boosting
- XGBoost
- AdaBoost
- LightGBM

## Key Features

- Comprehensive data preprocessing and visualization
- Implementation of multiple machine learning models
- Advanced visualization techniques (UMAP)
- Detailed model evaluation and comparison
- Hyperparameter tuning for each model
- Feature importance analysis

## Installation

To set up the project environment:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sonar-rock-vs-mine-prediction.git
   ```
2. Navigate to the project directory:
   ```
   cd sonar-rock-vs-mine-prediction
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the project:

1. Open the Jupyter notebook:
   ```
   jupyter notebook Sonar_rock_vs_mine_prediction_using_multi_model.ipynb
   ```
2. Run the cells in the notebook sequentially to perform data analysis, model training, and evaluation.

## Results

The project compares the performance of various models based on accuracy, precision, recall, and F1-score. Here are the results of our model comparison:

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 0.7857   | 0.7889    | 0.7857 | 0.7869   |
| Decision Tree         | 0.6667   | 0.7222    | 0.6667 | 0.6697   |
| Random Forest         | 0.7857   | 0.7990    | 0.7857 | 0.7883   |
| K-Nearest Neighbors   | 0.7857   | 0.8349    | 0.7857 | 0.7883   |
| Support Vector Machine| 0.9048   | 0.9107    | 0.9048 | 0.9057   |
| Neural Network        | 0.8333   | 0.8459    | 0.8333 | 0.8354   |
| Gradient Boosting     | 0.7857   | 0.8349    | 0.7857 | 0.7883   |
| XGBoost               | 0.8333   | 0.8617    | 0.8333 | 0.8357   |
| AdaBoost              | 0.8333   | 0.8459    | 0.8333 | 0.8354   |
| LightGBM              | 0.8333   | 0.8459    | 0.8333 | 0.8354   |

Based on these results, the Support Vector Machine (SVM) model performs the best with the highest accuracy, precision, recall, and F1-score.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The Sonar dataset used in this project is from the UCI Machine Learning Repository.
- Thanks to the scikit-learn, XGBoost, and LightGBM teams for their excellent machine learning libraries.
