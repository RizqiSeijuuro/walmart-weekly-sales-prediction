<h1 align='center'>Weekly Sales Prediction at Walmart Dataset</h1>

<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/c6fe15945e635d8b367f795f9f8e85b89a50d100/images/Walmart.jpg" width=600>
</p>

## Business Case

Walmart is one of the largest multinational retail companies in the world. Walmart has many competitors in the retail sector, so strategic decisions are needed in order to maintain its position.

By using exploratory data analysis, on past data from Walmart, I created a model that can predict Weekly Sales in Walmart Store. This model can provide information that helps business managers identify and understand weaknesses in business planning. 

Email: rizqiansyah52@gmail.com <br>
LinkedIn: www.linkedin.com/in/muhammad-rizqiansyah <br>

## Table of Contents
<details open>
<summary>Show/Hide</summary>
<br>

1. [ File Descriptions ](#File_Description)
2. [ Technologies Used ](#Technologies_Used)    
3. [ Structure ](#Structure)
4. [ Executive Summary ](#Executive_Summary)
   * [ 1. Early EDA and Cleaning](#Early_EDA_and_Cleaning)
   * [ 2. Feature Selection ](#Feature_Selection) 
   * [ 3. Modelling, Hyperparameter Tuning & Evaluation](#Modelling)
       * [ Conclusion ](#Conclusion)
       * [ Future Improvements ](#Future_Improvements)
</details>

## File Descriptions
<details>
<a name="File_Description"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>[ Data ](https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/tree/main/data)</strong>: folder containing all data files
    * <strong>1.Walmart.csv</strong>: data before any changes
    * <strong>2.data.csv</strong>: data after cleaning, feature engineering and feature elimination (no dropping outliers)
    * <strong>3.data_dropped_outlier.csv</strong>: data after cleaning, feature engineering and feature elimination (with dropped outliers)
    * <strong>3.data_tf_outlier.csv</strong>: data after cleaning, feature engineering and feature elimination (with outliers transformation)
    * <strong>4.data_enc.csv</strong>: data after cleaning, feature engineering and feature elimination (no dropping outliers & encoding)
    * <strong>4.data_drop_outlier_enc.csv</strong>: data after cleaning, feature engineering and feature elimination (with dropped outliers & encoding)
    * <strong>1.data_tf_outlier_enc.csv</strong>: data after cleaning, feature engineering and feature elimination (with outliers transformation & encoding)
    * <strong>2.X.csv</strong>: data with x values from preprocessed dataset (no dropping outliers & encoding)
    * <strong>3.y.csv</strong>: data with y values from preprocessed dataset (no dropping outliers & encoding)
    * <strong>3.X_2.csv</strong>: data with x values from preprocessed dataset (with dropped outliers & encoding)
    * <strong>4.y_2.csv</strong>: data with y values from preprocessed dataset (with dropped outliers & encoding)
    * <strong>4.X_3.csv</strong>: data with x values from preprocessed dataset (with outliers transformation & encoding)
    * <strong>4.y_3.csv</strong>: data with y values from preprocessed dataset (with outliers transformation & encoding)
* <strong>[ Images ](https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/tree/main/images)</strong>: folder containing images used for README and presentation pdf
* <strong>[ 1.Early_EDA.ipynb ](https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/c6fe15945e635d8b367f795f9f8e85b89a50d100/1.Early_EDA.ipynb)</strong>: notebook with early data exploration and data manipulation
* <strong>[ 2.Feature_Selection.ipynb ](https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/c6fe15945e635d8b367f795f9f8e85b89a50d100/2.Feature_Selection.ipynb)</strong>: notebook with feature selection
* <strong>[ 3.Modelling_Tuning___Evaluation.ipynb ](https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/c6fe15945e635d8b367f795f9f8e85b89a50d100/3.Modelling_Tuning___Evaluation.ipynb)</strong>: notebook with all the models created, final model selection, testing, and model evaluation
* <strong>[ Weekly_Sales_Prediction_Presentation.pdf ](https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/feb18f0171f80290d61f324c7f93691e51894bf9/Weekly_Sales_Prediction_Presentation.pdf)</strong>: presentation summarising project case, processes, and findings
</details>

## Tecnologies Used:
<details>
<a name="Technologies_Used"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>Python</strong>
* <strong>Pandas</strong>
* <strong>Numpy</strong>
* <strong>Matplotlib</strong>
* <strong>Seaborn</strong>
* <strong>Scikit-Learn</strong>
</details>

## Structure of Notebooks:
<details>
<a name="Structure"></a>
<summary>Show/Hide</summary>
<br>
    
1. Early EDA and Data Cleaning
   * 1.1 Imports
   * 1.2 Data Understanding
   * 1.3 Checking for Nulls
   * 1.4 Check Duplicated Data
   * 1.5 Feature Engineering: Date Variable
   * 1.6 Data Distribution
     * 1.6.1 Data Distribution: Target
     * 1.6.2 Data Distribution: Numerical Features
     * 1.6.3 Data Distribution: Categorical Features
   * 1.7 Feature Analysis
     * 1.7.1 Feature Analysis: Holiday_Flag vs Weekly_Sales
     * 1.7.2 Feature Analysis: Day, Month, Year vs Weekly_Sales
     * 1.7.3 Feature Analysis: Unempoloyment, CPI vs Weekly_Sales
   * 1.8 Finding and Handling Outliers
   * 1.9 Data Encoding
   * 1.10 Exporting Data

2. Feature Selection
   * 2.1 Imports
   * 2.2 Feature Importance
   * 2.3 Feature Ranking using RFE (Recursive Feature Elimination)
   * 2.4 Feature Decomposition using PCA (Principal Component Analysis)
   * 2.5 Exporting Data

3. Modelling & Evaluation
   * 3.1 Imports
   * 3.2 Create Function
   * 3.3 Modelling
     * 3.3.1 Linear Regression
       * 3.3.1.1 Dataset 1 (With Outliers)
       * 3.3.1.2. Dataset 2 (Outliers Dropped)
       * 3.3.1.3 Dataset 3 (Outliers Transformed)
       * 3.3.1.4 Evaluation for Linear Regression
     * 3.3.2 Ridge Regression
       * 3.3.2.1 Ridge Regression: Base Model
       * 3.3.2.2 Ridge Regression: Hyperparameter Tuning
     * 3.3.4 ElasticNet Regression
</details>  
   
<a name="Executive_Summary"></a>
## Executive Summary


<a name="Early_EDA_and_Cleaning"></a>
### Early EDA and Data Cleaning
<details open>
<summary>Show/Hide</summary>
<br>

#### Data Cleaning
Walmart dataset consisting of 6435 rows and 8 columns, which has the target column Weekly_Sales. we could say the dataset is quite clean, where there are no missing values and duplicate values.

On holidays from 2010-2012, all of them are in the dataset except Thanksgiving and Christmas, there is no data in 2012. This is due to the incomplete data range. The Data ranges is from 5 February 2010 - 6 October 2012.

<h5 align="center">Walmart Dataset Information</h5>
<table><tr><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/302735cece83b42514877e9e3485844742ef6a76/images/Slide1.JPG' width=500></td><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide2.JPG' width=500></td></tr></table>

There are outliers in the Temperature column as many as 3 rows. Our assumption is that winter starts in early December and ends at the end of February 2011. So, these outliers can be considered reasonable. Therefore, we will leave this outlier. The outliers in the Unemployment column are almost 500 rows. Apparently, the low unemployment rate occurred in 2012, while the high one occurred throughout the year. We cannot provide strong enough assumptions so that further outlier handling is necessary.

<h5 align="center">Outliers (Temperature Column (Left) & Unemplyment Column (Right))</h5>
<table><tr><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide3.JPG' width=500></td><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide4.JPG' width=500></td></tr></table>

so that there will be 3 scenarios used, (1) leave the outliers (2) outliers removed (3) outlier transformation.
<h5 align="center">Handling Outliers</h5>
<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide5.JPG" width=600>
</p>

#### Exploratory Data Analysis

<h5 align="center">Annual Sales Graph</h5>
<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide6.JPG" width=600>
</p>

From the Annual Sales graph, it can be seen that the largest sales occurred in 2012, followed by 2010 and 2011. It should be noted that the data ranges are not balanced every year, so we only take the range from February to July to find out the difference.

<h5 align="center">Average Weekly Sales in Months</h5>
<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide7.JPG" width=600>
</p>

We dig deeper. It turns out that the largest Average Weekly Sales is in December. Where in that month there are Christmas and New Year's holidays, followed by November where in that month there is a Thanksgiving holiday.

<h5 align="center"Weekly Sales vs Time</h5>
<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide8.JPG" width=600>
</p>

If we dig deeper, we can see that the biggest Weekly Sales occur around November and December. This strengthens our assumption that November to December is indeed the highest demand for goods.

<h5 align="center">Average Weekly Sales on Holiday vs Normal Day</h5>
<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide9.JPG" width=600>
</p>

Average Weekly Sales when Holiday is higher on weekdays. Of course, but there is a unique thing that I will explain later.

<h5 align="center">Average Weekly Sales on Holiday vs Normal Day (detailed)</h5>
<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide10.JPG" width=600>
</p>

If we dig deeper, Thanksgiving has the highest increase in Average Weekly Sales compared to other holidays. Followed by the Super Bowl with a slight difference. The unique thing here is that sales during Labor Day and Christmas are lower than normal days.

<h5 align="center">Average Weekly Sales on 1 week before Holiday vs Normal Day (detailed)</h5>
<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide11.JPG" width=600>
</p>

Meanwhile, at 1 week before a holiday. It turned out that Christmas experienced the highest increase in Average Weekly Sales compared to other holidays. Followed by the Super Bowl with a slight difference. Again, the unique thing here is that Thanksgiving and Labor Day are lower than normal days.

<h5 align="center">Weekly Sales vs Unemployment Rate (Left) & CPI (Right)</h5>
<table><tr><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide12.JPG' width=500></td><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide13.JPG' width=500></td></tr></table>

For the Unemployment Rate factor, the insight that we can take here is that the highest Average Weekly Sales is at a very low Unemployment Rate and for the Customer Price Index (CPI) factor, the highest Average Weekly Sales is at a slightly low CPI level. From these 2 variables, it means that the country's economic factors here do not really affect Walmart's weekly sales.


</details>  

<a name="Feature_Selection"></a>
### Feature Selection
<details open>
<summary>Show/Hide</summary>
<br>
    
Here are RFE and PCA. RFE works by selecting features based on an estimator recursively, the estimator we use is the Linear Regression coefficient. Meanwhile, PCA works by reducing features by combining based on the value of the variance.

<h5 align="center">Recursive Feature Elimination (Left) & Principal Componenet Analysis (Right)</h5>
<table><tr><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide14.JPG' width=500></td><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide15.JPG' width=500></td></tr></table>

For RFE, we plot the value of R2 along with the loss of features in the dataset. We can see that the R2 value begins to fall drastically after the number of features dropped exceeds 75, so the best number of features is the total number of features 151-75, which is 76. For PCA, we plot as well as the previous RFE. Unfortunately PCA is overfitting because the R2 value is very good on the training dataset, but it is negative and fluctuates drastically in the testing dataset. So it was decided to use an RFE whose performance is more stable.

</details>

<a name="Modelling, Hyperparameter Tuning & Evaluation"></a>
### Modelling:
<details open>
<summary>Show/Hide</summary>
<br>

Here we divide the dataset by a ratio of 60:40. There are 4 models tested here, namely Linear Regression, Ridge Regression, Lasso Regression, and ElasticNET Regression. Both of these models assume a linear relationship between features and targets..

For the evaluation metric, there is R2 to see how well the model that has been made predicts the data, we use this metric because this metric is often used and also we did not find any problems when using R2. MAE, MSE, and RMSE are the most frequently used metrics, but these three metrics are very scale-dependent which does not match our dataset, because sales data can range in the millions. MAPE and NRMSE are scale-independent, but MAPE is asymmetrical and gives a large error value if the resulting value is negative, so we decided to use NRMSE, which is RMSE divided by the standard deviation of the prediction.

<h5 align="center">Model Used (Left) & Evaluation Metric Used (Right)</h5>
<table><tr><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide16.JPG' width=500></td><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide17.JPG' width=500></td></tr></table>

In Linear Regression, we tried 3 data with the scenarios previously mentioned, then we also applied 3 feature scaling methods, it is standardization, normalization, and robust scaler. Based on the results of the best R2 value, we can see from the thick line and large dot size, all of which are almost owned by the second dataset, which is where all outliers are dropped. And for the smallest Normalized RMSE, it is again almost owned by the second dataset, which is where all outliers are dropped. It was therefore decided to use a second dataset to compare the models.

<h5 align="center">Linear Regression: R2 (Left) & NRMSE (Right)</h5>
<table><tr><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide18.JPG' width=500></td><td><img src='https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide19.JPG' width=500></td></tr></table>

Here are the results for Ridge Regression, the base model on the left, then the model with hyperparameter tuning on the right. The parameter we're tuning for is the alpha, which governs how much Ridge Regression penalizes the weight value. From the plot, we can see that there is a slight improvement in the model that has been tuning parameters but not too significant.

<h5 align="center">Ridge Regression: R2 & NRMSE</h5>
<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide20.JPG" width=600>
</p>

Meanwhile, for lasso regression the Base model has very poor performance on data with feature scaling. Then after we tuning the alpha, the performance changed to be much better, but this happened because we changed the alpha to be close to zero, which according to the documentation is not recommended because Lasso with alpha 0 is the same as Linear Regression.

<h5 align="center">Lasso Regression: R2 & NRMSE</h5>
<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide21.JPG" width=600>
</p>

Meanwhile, the ElasticNET regression Base model has a much worse performance. Our assumption is that because ElasticNet is a combination of Lasso and Ridge, since Lasso from ScikitLearn doesn't seem to be able to handle the problem we have, we assume ElasticNet might have the same problem.

<h5 align="center">ElasticNET Regression: R2 & NRMSE</h5>
<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide22.JPG" width=600>
</p>

I focused on 3 factors of defining a good model:

1. Good Validation Accuracy
2. Good Training Accuracy
3. Small Difference between Training and Validation Accuracy

From the performance comparison between models, it can be seen that the results between Linear Regression and Ridge Regression are quite competitive here, so we decided to use Ridge Regression because the penalty function in ridge regression can avoid the possibility of overfitting in the future.

<h5 align="center">Comparison Between Models</h5>
<p align="center">
  <img src="https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/main/images/Slide23.JPG" width=600>
</p>

</details>

<a name="Conclusion"></a>
#### Conclusion

- The best model for predicting weekly sales is Ridge Regression. This model can provide information that helps business managers identify and understand weaknesses in business planning.
- The Marketing Division should increase advertising during the weeks after Christmas and before Thanksgiving.
- Need additional people from the Logistics division in November-December because sales increased significantly compared to other months.
- Re-stock goods sufficiently on weekdays to minimize production costs.

</details>

<a name="Future_Improvements"></a>
#### Future Improvements/Development

- Model using random forest regressor - see if better accuracy can be achieved
- Create a working application to test Weekly Sales written by people
- Try a different pre-processing approach and see if model performances change
- Bring in new sources of data to see if there are significant differences on time factor used

</details>
