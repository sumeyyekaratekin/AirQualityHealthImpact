# Air Quality and Health Impact Analysis

This project aims to analyze the impact of air quality on health using the dataset obtained from Kaggle.

## Dataset

The dataset contains various measurements related to air quality and health impacts. You can access the dataset: [Air Quality and Health Impact Dataset](https://www.kaggle.com/datasets/rabieelkharoua/air-quality-and-health-impact-dataset).

## Dataset Preparation
For testing the dataset, the random state value has been set to 58. This ensures that the dataset is split in the same way each time it is run.

### Importing Libraries and Dataset
The necessary libraries are imported, and the dataset is loaded for analysis.

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Data Exploration and Cleaning

- **Data Overview**: Inspect the initial rows and summary statistics of the dataset.
- **Data Types and Conversion**: Convert the `HealthImpactClass` column to integer type.
- **Data Distribution**: Visualize the distribution of health impact classes.
- **Correlation Analysis**: Identify correlations between different columns.

### Feature Engineering

- **Impact Classification**: Define a function to categorize health impact based on `HealthImpactScore`.

### Data Splitting

- **Train-Test Split**: Divide the data into training and testing sets.

### Data Scaling

- **Normalization**: Scale features using `MinMaxScaler`.

## Model Training and Evaluation
The following machine learning models were utilized in this project:

- **Artificial Neural Network (ANN)**
- **CatBoost Classifier**
- **Support Vector Machine (SVM)**
- **LightGBM**
- **XGBoost**
- **Logistic Regression**
- **Random Forest Classifier**
- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN)**
- **Bernoulli Naive Bayes**
- **Gaussian Naive Bayes**
- **Multinomial Naive Bayes**


## Code Description
The project is contained within a single Jupyter Notebook file:

`AirQualityHealthImpact.ipynb` : This notebook includes code for data cleaning, preprocessing, analysis, visualization, and modeling.

## Results

### Accuracy of Different Models
Based on the analysis, the accuracy of various models is as follows:

- **ArtificialNeuralNetwork**: 96.95%
- **CatBoostClassifier**: 96.90%
- **SupportVectorMachine**: 95.63%
- **LightGBM**: 95.39%
- **XGB**: 95.09%
- **LogisticRegression**: 95.01%
- **RandomForestClassifier**: 94.63%
- **DecisionTreeClassifier**: 92.19%
- **KNeighborsClassifier**: 81.07%
- **BernoulliNB**: 86.24%

### Summary
- The **ArtificialNeuralNetwork** model achieved the highest accuracy at 96.95%.
- **CatBoostClassifier** and **SupportVectorMachine** also showed high performance.
- **KNeighborsClassifier** and **BernoulliNB** had lower accuracy compared to other models.

These results can be used to compare model performance and select the best-performing model.

## Usage
To use the notebook:
1. Install the required libraries.
2. Download the dataset and include it in the project directory.
3. Open the `AirQualityHealthImpact.ipynb` file using Jupyter Notebook or JupyterLab.
4. Execute the cells in the notebook to run the analysis and modeling tasks.
