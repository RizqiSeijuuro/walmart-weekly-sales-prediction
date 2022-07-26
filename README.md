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
   * [ 2. Further EDA and Preprocessing ](#Further_EDA_and_Preprocessing) 
   * [ 3. Modelling and Hyperparameter Tuning ](#Modelling)
   * [ 4. Evaluation ](#Evaluation)
       * [ Future Improvements ](#Future_Improvements)
</details>

## File Descriptions
<details>
<a name="File_Description"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>[ Data ](https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/tree/main/data)</strong>: folder containing all data files
    * <strong>1.Walmart.csv</strong>: data before any changes
    * <strong>2.Walmart_structured.csv</strong>: data after cleaning, feature engineering and feature elimination
    * <strong>3.x_train_data.csv</strong>: training data with x values from preprocessed dataset
    * <strong>3.y_train_data.csv</strong>: training data with y values from preprocessed dataset
    * <strong>4.x_test_data.csv</strong>: test data with x values from preprocessed dataset
    * <strong>4.y_test_data.csv</strong>: test data with y values from preprocessed dataset
* <strong>[ Images ](https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/tree/main/images)</strong>: folder containing images used for README and presentation pdf
* <strong>[ 1.Early_EDA.ipynb ](https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/c6fe15945e635d8b367f795f9f8e85b89a50d100/1.Early_EDA.ipynb)</strong>: notebook with early data exploration and data manipulation
* <strong>[ 2.Feature_Selection.ipynb ](https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/c6fe15945e635d8b367f795f9f8e85b89a50d100/2.Feature_Selection.ipynb)</strong>: notebook with all the models created
* <strong>[ 3.Modelling_Tuning___Evaluation.ipynb ](https://github.com/RizqiSeijuuro/walmart-weekly-sales-prediction/blob/c6fe15945e635d8b367f795f9f8e85b89a50d100/3.Modelling_Tuning___Evaluation.ipynb)</strong>: notebook with all the models created, final model selection, testing, and model evaluation
* <strong>[ Weekly_Sales_Prediction_Presentation.pdf ](https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Hilton_Hotel_Presentation.pdf)</strong>: presentation summarising project case, processes, and findings
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
#### Early EDA and Data Cleaningg: 

The initial shape of the dataset was (35078,5). The 5 columns was as expected, but there were double the number of rows as the number of reviews scraped. There were null rows with only hotel_name and no other values, so I removed those rows, bringing us back to the expected 17538.

This project entailed the use of classification models, and for reliable results, I had to remove reviews to undo class imbalance. Using this visualisation I saw that were much less reviews with a score of 1 compared to reviews with a score of 3, 4, and 5. To combat this imbalance, I randomly removed reviews with scores of 2, 3, 4, and 5, to match with 1 (1881 reviews). 

<h5 align="center">Histogram of Scores for All Hotels (With  Class Imbalance (Left) vs Without  Class Imbalance (Right))</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/histogram_of_scores_for_all_hotels.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/histogram_of_scores_for_all_hotels_after_balancing.png' width=500></td></tr></table>

I combined the review p1 and review p2 column into one to make future vectorisation much easier, then I saved the cleaned dataset as a csv, for the next stage.
</details>  

<a name="Further_EDA_and_Preprocessing"></a>
### Feature Selection
<details open>
<summary>Show/Hide</summary>
<br>
    
The cleaned dataset had a shape of (9405,4). I started with some analysis on the text columns; review and review summary.

Using the FreqDist function in the ntlk library I plotted a graph with the most frequent words and phrases in both columns. Stopwords were removed to capture the more meaningful words.

<h5 align="center">Distribution Plot of Frequent Words and Phrases in Text ( Review Summary (Left) and Review (Right) )</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/freq_dist_review_sum.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/freq_dist_review.png' width=500></td></tr></table>

I had noticed a lot of the most frequent words in the review text happened to be words with no sentimental impact, so I iteratively removed unmeaningful words such as 'room', 'hotel', 'hilton' etc. I did this as a precaution, as some of these words may impact my model accuracies.

<h5 align="center">World Cloud of Frequent Words and Phrases in Text After Removing Unmeaningful Words ( Review Summary (Left) and Review (Right) )</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/word_cloud_review_sum.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/word_cloud_review.png' width=500></td></tr></table>

To narrow down the feature words I applied stemmation and lemmitisation to both the reviews and review summaries. 

<h5 align="center">Example of Lemmatisation and Stemmation Applied to a Review and Review Summary</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/lemm_stemm_ex.png" width=600>
</p>

Stemmation had broken down some words into words that don't exist, whereas lemmitisation had simplified adjectives and verbs to their root form. I chose to continue with the lemmitised version of the texts for further processing.

Prior to vectorising the current dataset, I did a train, test split to save the test data for after modelling.

Using the lemmed texts for review and review summary I used TF-IDF vectorisation with an ngram range of 2, leaving me with a vectorised dataset with 138 words and phrases (112 from reviews and 26 from review summaries). I then saved the x and y train data in separate csv files for modelling.
</details>

<a name="Modelling"></a>
### Modelling:
<details open>
<summary>Show/Hide</summary>
<br>

I have created .py files; Classifiction.py and Ensemble.py with classes, that contain functions to simplify the modelling process, and to neaten up the modelling notebook.

I did another data split into Train and Validation data in preparation for using GridSearch Cross Validation. I also chose Stratified 5-fold has a my choice for cross validating.

For the majority of models I created, I applied hyperparameter tuning, where I started with a broad range of hyperparameters, and tuned for optimal train accuracy and validation accuracy. 


<h5 align="center">Table Comparing Best Models</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/all_models.png" width=600>
</p>

Initially, I thought the validation accuracy was low for most of the models I created, but when considering these models were attempting to classify for 5 different classes, 0.45 and greater seems very reasonable (where 0.2 = randomly guessing correctly).

I have saved all the models using the pickle library's dump function and stored them in the Models folder.
</details>

<a name="Evaluation"></a>
### Evaluation
<details open>
<summary>Show/Hide</summary>
<br>

I focused on 3 factors of defining a good model:

1. Good Validation Accuracy
2. Good Training Accuracy
3. Small Difference between Training and Validation Accuracy

I chose the Stacking ensemble model ( (Adaboost with log_reg_2) stacked with log_reg_2 ) as my best model, because it has the highest validation accuracy with only around 3.5% drop from train to validation in accuracy. I wanted to minimise overfitting and make the model as reusable as possible. Stacking achieved a reasonable training accuracy as well, although it did not reach the level of some of the other ensemble techniques.

I next tested the best model with the earlier saved test data. The model managed to get a high test accuracy, similar to the validation data from the model training stage. This is very good, proving that prioritising a high validation score, and minimising the difference between train and validation accuracy, has helped it classify new review texts very well.

<h5 align="center">Test Results</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/test_results.png" width=600>
</p>

Looking at the precision, recall, and f1 score, I also noticed the scores were higher around scores of 1 and 5, lower for 2, 3, and 4. This shows that the models performs well on more extreme opinions on reviews than mixed opinions.

Looking into different metrics and deeper into my best model; Stacking, I learnt that most the False Postives came from close misses (e.g. predicting a score of 4 for a true score of 5). This is best shown by these two confusion matrixes (validation and test). 

<h5 align="center">Confusion Matrix for Validation and Test Data Predictions ( Validation (Left) and Test (Right) )</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/validation_conf_matrix.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/test_conf_matrix.png' width=500></td></tr></table>

The adjacent squares of the diagonal going across the confusion matrix, shows that the model's second highest prediction for a given class (review score) is always a review score that is +-1 the true score.
Very few reviews that have a score of 5, have been predicted to have a score of 1 or 2. This is very relieving to know, the majority of the error for the model, is no different to the error a human may make classifying a review to a score with a scale of 1-5.

- most errors were near misses (e.g. 5 predicted as 4)
- extreme scores (1 and 5) were relatively accurate
- comparable to human prediction
- reusable and consistent


Given the classifcation problem is 5 way multi-class one and the adjacent classes can have overlap in the english language even to humans, this model I have created can be deployed.

Applying this model will address the problem of not having a full understanding of public opinion of our hotel. We can apply this to new sources for opinions on our hotel and yield more feedback then we did had before.

<a name="Future_Improvements"></a>
#### Future Improvements

- Model using neural networks - see if better accuracy can be achieved
- Create a working application to test new reviews written by people
- Try a different pre-processing approach and see if model performances change
- Bring in new sources of data to see if there are significant differences on frequent words used

</details>

#### Future Development
    
* Create a webscraper spider for twitter, reddit, etc for further model assessment
    
</details>
