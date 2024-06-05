# Practical Application III: Comparing Classifiers
### Johennie Helton
#### June, 2024



In this practical application, the goal is to compare the performance of 
K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines classifiers. <br>
Utilizing a dataset related to marketing bank products over the telephone.<br>

Notebook: [https://github.com/johennie/bank_marketing/notebooks/jh_prompt_III.ipynb](https://github.com/johennie/bank_marketing/blob/main/notebooks/jh_prompt_III.ipynb)
<br>Data: [https://github.com/johennie/bank_marketing/data/bank-additional-full.csv](https://github.com/johennie/bank_marketing/blob/main/data/bank-additional-full.csv)

## 1. Business Understanding
The business objective is to identify if a client chooses (target == yes / 1) a long-term deposit after a marketing campaing. In order to do so, we want to identify a predictive model that classifies the target data.



## 2. Data Understanding

<br>The data is collected from a Portuguese bank's 17 direct marketing campaing which occurred between May 2008 and November 2010. The data attributes were obtained from the campaign related to contact information, and were augmented with internal bank data (for client's characteristics)
<br>
<br>During this phase we are to describe and explore the data to make sure it can be used for analysis and visualization in understanding if the client subscribed to a term deposit.
<br>The data has 41176 entries with 21 features, and it is found in the data/bank-additional-full.csv file which contains the following columns:
<br>    age               - numeric
<br>    job               - type of job, categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'
<br>    marital           - marital status, categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed
<br>    education         - categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'
<br>    default           - has credit in default? categorical: 'no','yes','unknown'
<br>    housing           - has housing loan? categorical: 'no','yes','unknown'
<br>    loan              - has personal loan? categorical: 'no','yes','unknown'
<br>    contact           - contact communication type, categorical: 'cellular','telephone'
<br>    month             - last contact month of year, categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec'
<br>    day_of_week       - last contact day of the week, categorical: 'mon','tue','wed','thu','fri'
<br>    duration          - last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
<br>    campaign          - number of contacts performed during this campaign and for this client (numeric, includes last contact)
<br>    pdays             - number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
<br>    previous          - number of contacts performed before this campaign and for this client (numeric)
<br>    poutcome          - outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
<br>    emp.var.rate      - employment variation rate - quarterly indicator (numeric)
<br>    cons.price.idx    - consumer price index - monthly indicator (numeric)
<br>    cons.conf.idx     - consumer confidence index - monthly indicator (numeric)
<br>    euribor3m         - euribor 3 month rate - daily indicator (numeric)
<br>    nr.employed       - number of employees - quarterly indicator (numeric)
<br>    y                 - target. has the client subscribed to a term deposit? (binary: 'yes','no')
<br>
The following plot demonstrates that the outcome (target) has a skewed distribution toward no subscription to a term deposit which affect the performance and reliability of models. <br>
<br>term_deposit_subscription
<br>0     36537
<br>1      4639

![result_pieplot.png](images%2Fresult_pieplot.png)

## 3. Data Preparation and Exploration
<br>We checked for null values, missing values, and duplicate rows. Also, notice that the target feature is imbalanced.
<br>Created a new featured called 'term_deposit_subscription' with the encoded target values of yes = 1 and no to 0 
<br>Also divided the age into bins like this:
<br>    if age < 20: age_less_than_20_yrs
<br>    if 20 <= age < 40: age_between_20_and_40_yrs
<br>    if 40 <= age < 60: age_between_40_and_60_yrs
<br>    if 60 <= age < 80: age_between_60_and_80_yrs
<br>    if 100 <= age : age_above_100_yrs
<br>The features identified for modeling were: job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome, age, duration, campaign, pdays, and previous with the target being the newly created term_deposit_subscription.
<br>These two plots depict the age group that subscribed to a term deposit (== 1) by job (on the left) and by eduction (on the right). Therefore, it seems that age has an impact on the outcome.

![byJob_byEducation.png](images%2FbyJob_byEducation.png)

<br>We separated the numerical and categorical features, and created a pipeline for each to scale or encode the data. Then, split the data into training (80%) and testing (20%) sets. 
<br>Since the target data is imbalanced, we also want to compare the performance of the models with a more balanced data set. Therefore, we sample the data so that the subscription to a term deposit has the same number of yes and no. 

<br>undersampled_distribution
<br>yes    4639
<br>no     4639


## 4. Modeling

At first, we selected a RandomForestClassifier model as a base model to establish a baseline; the selection is due to performing well on categorical and numerical data. 
<br>Then, we used Logistic Regression to build a basic model which is efficient and performs well on binary classification. We also used this model to identify the coefficient of each feature and visualize the strength and their direction on the prediction.
<br>After that, we build KNN,Decision Tree, and SVM to compare their performance. Also, we do some feature exploration with age categories, and do some hyperparameter tuning. 
<br>Finally, we compare all those models with models using the undersampled distribution data set.

<br> That is the models include:
RandomForestClassifier<br>
LogisticRegression<br>
K-Nearest Neighbors (KNN)<br>
Decision Tree<br>
Support Vector Machine (SVM)<br>
The models were evaluated under different conditions:<br>
Default parameters on the original dataset.<br>
Undersampling of the majority class.<br>
Grid Search for hyperparameter tuning on both the original and undersampled datasets.<br>

<br>Key Metrics
<br>Train Time: Time taken to train the model.
<br>Train Accuracy: Accuracy on the training data.
<br>Test Accuracy: Accuracy on the testing data.
<br>F1 Score: Harmonic mean of precision and recall.
<br>Best Params: Best parameters obtained from Grid Search.
<br>

### Results
<br>Default parameters on original data:
<br>- RandomForestClassifier: has high train accuracy (1.0) indicating possible overfitting.
<br>- LogisticRegression: has good performance with train accuracy of 0.909, and low train time.
<br>- KNN: performs well with train accuracy of 0.924, and moderate train time.
<br>- Decision Tree: similar to the RandomForest it has high train accuracy (1.0) indicating overfitting, and the train time is relatively low.
<br>- SVM: has good performance with train accuracy of 0.918, but it has a high train time.

Undersampled Data:
<br>- RandomForestClassifier: again we see a high train accuracy (1.0) indicating overfitting, notice the reduced train time compared to the original data.
<br>- LogisticRegression: the metrics are in the low 0.80s, and it has very low train time.
<br>- KNN: train accuracy is 0.866, but the test accuracy and F1 score are in the high 0.7s, it also has very low train time.
<br>- Decision Tree: it has high train accuracy (1.0) indicating overfitting, 
with lower test accuracy and F1 scores; though, it has low train time.
<br>- SVM: train accuracy, test accuracy, and F1 score are in the mid 0.8s with moderate train time.

Grid Search Hyperparameter Tuning:
<br> - RandomForestClassifier: the grid search on original data show similar performance to default parameters 
with slight improvements in some metrics. However, grid search on undersampled data indicates potential underfitting due to the reduced 
train and test accuracy.
<br> - LogisticRegression: the grid search slightly improves the F1 score on both original and undersampled data.
<br> - KNN: the grid search slightly improves performance metrics, but the improvements are not substantial.
<br> - Decision Tree: the grid search improves F1 score on the original data but shows underperformance on undersampled data.
<br> - SVM: the grid search shows negligible improvements in metrics on original data but does not significantly 
impact undersampled data performance.

![model_performance.png](images%2Fmodel_performance.png)

![summary_results.png](images%2Fsummary_results.png)

## 5. Evaluation

We evaluated the performance of various models the best performing model is LogisticRegression with GridSearch on the original dataset
with good balance of performance metrics and training time.<br>
RandomForestClassifier and Decision Tree models show signs of overfitting in both original and undersampled data. We expected that 
undersampling the majority target class would improve metrics, but instead it lowered the test accuracy and F1 score. 
<br>This indicates that undersampling is not effective in this dataset. Using GridSearch increased metrics but the gains may not justify the increase on
train time. This is obvious with SVM model.

## 6. Next Steps
Undersampling oversimplifies the problem assuming that balancing the classes by reducing the majority class is sufficient  
but in this dataset did not. Therefore, experimenting with more advanced resampling techniques to address class imbalance 
more effectively may yield better results. Also, conducting more extensive hyperparameter tuning to
find better parameter combinations with less computational cost may provide a better performing model. 
