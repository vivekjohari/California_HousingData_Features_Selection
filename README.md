
![Logo](https://github.com/vivekjohari/California_HousingData_Features_Selection/blob/main/California_Housing_Price_Predictor.png)


# California housing price prediction using Linear Regression model

This Project is created to include various the Feature Selection Techniques like Filter based methods, Wrapper methods, Embedded Methods and Dimensionality Reduction Techniques  like Principal Component Analysis (PCA) while building the regression model . It helps in reducing the complexity of the model (number of features) and improve the efficiency of the models by removing the features which has less impact on the outcome.

## Implementation Details

- Dataset: California Housing Dataset (view below for more details)
- Model: [Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Input: 8 features - Median Houshold income, House Area, ...
- Output: House Price

## Dataset Details

This dataset was obtained from the StatLib repository ([Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html))

This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the sklearn.datasets.fetch_california_housing function.

- [California Housing Dataset in Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- 20640 samples
- 8 Input Features: 
    - MedInc median income in block group
    - HouseAge median house age in block group
    - AveRooms average number of rooms per household
    - AveBedrms average number of bedrooms per household
    - Population block group population
    - AveOccup average number of household members
    - Latitude block group latitude
    - Longitude block group longitude
- Target: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)

## Evaluation and Results
![alt text](https://github.com/123ofai/Demo-Project-Repo/blob/main/results/test.png)

As you can see from the above image, the model has signifcant amount of error in <x, y, z regions>

|Feature Selection method Used       | R2_score|MSE|
|------------------------------------|---------|----|
|SelectKBest                       |   0.57  |0.56|
|SelectPercentile                  |   0.57  |0.56|
|f_Regression                      |   0.57  |0.56|
|Correlation Techniques            |   0.56  |0.56|
|Recursive Feature Elimination (RFE)|   0.26 |0.97|
|Sequential Feature Selection       |   0.51 |0.64|
|Principal Component Analysis(PCA)  | 0.44   |0.74|


In the above results, we can see the value of R2_Score and MSE for the various features selection methods

## Key Takeaways

- We got to know about the various Feature Selection Techniques like Filter based methods, Wrapper methods, Embedded Methods and Dimensionality Reduction Techniques like Principal Component Analysis (PCA)


## How to Run

The code is built on Google Colab on an iPython Notebook. 

```bash
Simply download the repository, upload the notebook and dataset on colab, and hit play!
```


## Roadmap

What are the future modification you plan on making to this project?

- Try more models

- Try more Feature Selection Techniques


## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn


## FAQ

#### What is linear regression model?

Linear regression predicts the relationship between two variables by assuming a linear connection between the independent and dependent variables. It seeks the optimal line that minimizes the sum of squared differences between predicted and actual values. It can extend to multiple linear regression involving several independent variables and logistic regression, suitable for binary classification problems

#### What is Mean Squared Error (MSE)?

The Mean Squared Error (MSE) is an estimator measures the average of error squares i.e. the average squared difference between the estimated values and true value.

In MSE, we calculate the error by squaring the difference between the predicted value and actual value and averaging it across the dataset. 

MSE will never be negative since the errors are squared. The value of the error ranges from zero to infinity. MSE increases exponentially with an increase in error. A good model will have an MSE value closer to zero.


#### What is R2 score?
R2 score is used to evaluate the performance of a linear regression model. It is the amount of the variation in the output dependent attribute which is predictable from the input independent variable(s). 

Interpretation of R2 score:

Assume R2 = 0.58
It can be referred that 58% of the changeability of the dependent output attribute can be explained by the model while the remaining 42 % of the variability is still unaccounted for.

R2 indicates the proportion of data points which lie within the line created by the regression equation. A higher value of R2 is desirable as it indicates better results.

#### What is the California Housing Dataset?

California Housing dataset
--------------------------

**Data Set Characteristics:**

    :Number of Instances: 20640

    :Number of Attributes: 8 numeric, predictive attributes and the target

    :Attribute Information:
        - MedInc        median income in block group
        - HouseAge      median house age in block group
        - AveRooms      average number of rooms per household
        - AveBedrms     average number of bedrooms per household
        - Population    block group population
        - AveOccup      average number of household members
        - Latitude      block group latitude
        - Longitude     block group longitude

    :Missing Attribute Values: None

This dataset was obtained from the StatLib repository.
https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

The target variable is the median house value for California districts,
expressed in hundreds of thousands of dollars ($100,000).

## Acknowledgements

All the links, blogs, videos, papers you referred to/took inspiration from for building this project. 

 - [scikit-learn.org](https://scikit-learn.org/stable/user_guide.html)
 - [scikit-learn California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing)
 


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at askvivekjohari@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

