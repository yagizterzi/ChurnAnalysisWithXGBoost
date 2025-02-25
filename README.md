# Churn Analysis with XGBoost


This project applies XGBoost, a powerful gradient boosting algorithm, to analyze and predict customer churn. The goal is to identify factors influencing customer retention and optimize predictive performance.

## What is Churn and Why Do We Need to Analyze And Predict It

Churn refers to the rate at which customers leave or stop using a bank’s services. In the banking sector, understanding churn is crucial because losing customers not only reduces revenue but also increases costs—acquiring new customers often costs more than retaining existing ones.churn analysis and prediction are essential tools for banks to maintain customer relationships, optimize costs, improve service offerings, and ultimately sustain long-term profitability.




## Features
- Comprehensive Exploratory Data Analysis (EDA) to uncover churn patterns and insights
- Data preprocessing and feature engineering
- Model training and evaluation using XGBoost
- Detailed performance metrics and visualization

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yagizterzi/ChurnAnalysisWithXGBoost.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ChurnAnalysisWithXGBoost
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main script to perform EDA, train the model, and evaluate its performance:
  ```bash
  python fullproject.py
  ```
2. Run save_deploy file
    ```bash
    python save_deploy.py
    ```
3. Lastly run the test file
   ```bash
   python test_api.py
   ```

## Dependencies
Ensure the following libraries are installed:
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- seaborn

## Workflow
The project workflow consists of:
- **Exploratory Data Analysis (EDA):** Initial data analysis to understand churn patterns, identify trends, and inform feature selection.
- **Data Preprocessing & Feature Engineering:** Cleaning and transforming data to improve model performance.
- **Model Training & Evaluation:** Building the churn prediction model using XGBoost and assessing its performance using metrics such as accuracy, precision, recall, and F1-score.
- **Visualization:** Generating plots to visualize data insights and feature importance.

## Contributing
Feel free to fork this repository and submit pull requests for improvements.


## License
This project is licensed under the [License](LICENSE). See the LICENSE file for details.



