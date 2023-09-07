# Insurance_premium_prediction
The provided code appears to be an analysis and modeling pipeline for predicting insurance premiums. It seems to be a Jupyter Notebook (`.ipynb`) file that was originally created in Google Colab, as indicated by the comments at the top of the code.

Here's an overview of what the code does and its main components:

1. **Data Loading and Initial Exploration:**
   - The code starts by importing necessary libraries such as Pandas, NumPy, Seaborn, Plotly, Matplotlib, and others.
   - It loads insurance data from a CSV file located at "/content/insurance.csv" using Pandas and displays the first 5 rows using `data.head(5)`.
   - It checks for null values in the dataset using `data.info()` and `data.isnull().sum()`.
   - Basic statistics and information about the dataset are displayed using `data.describe().T` and `data.describe(include='object').T`.
   - Duplicates are removed using `data.drop_duplicates(inplace=True)`.

2. **Data Visualization:**
   - The code uses various visualization libraries (Seaborn, Plotly, Matplotlib) to explore the dataset.
   - A pair plot is created using Seaborn to visualize relationships between numeric variables, colored by the 'sex' column.
   - Histograms and Q-Q plots are generated to visualize the distribution of numeric features.
   - Age groups are created and plotted to analyze the age distribution of insured individuals.
   - Scatter plots are created to explore the relationship between age, expenses, region, and gender.
   - A linear regression model is used to visualize the relationship between age and expenses, with a distinction made for smokers.

3. **Correlation Analysis:**
   - A correlation heatmap is created using Seaborn to visualize the correlation between numeric features.

4. **Data Preprocessing:**
   - One-hot encoding is applied to categorical columns ('sex', 'smoker', 'region') to convert them into numerical format.

5. **Modeling:**
   - Several regression models are trained and evaluated on the dataset. These models include Linear Regression, Ridge Regression, Lasso Regression, Support Vector Regression, Decision Tree Regressor, Bagging Regressor, AdaBoost Regressor, Gradient Boosting Regressor, and XGBoost Regressor.
   - The code defines functions to evaluate model performance metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared (R²), and Adjusted R².
   - The models are trained on the dataset, predictions are made, and performance metrics are calculated and displayed for both training and test sets.

6. **Model Comparison:**
   - The code creates plots to compare the performance of different models using metrics like MSE, MAE, RMSE, R², and Adjusted R².

Overall, this code provides a comprehensive analysis of the insurance premium prediction problem, including data exploration, visualization, preprocessing, model training, and performance evaluation. It compares the performance of various regression models to determine which one performs best on the given dataset. The choice of the best model depends on the specific evaluation metric and requirements of the problem.
