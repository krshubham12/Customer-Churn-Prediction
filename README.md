# Telco Customer Churn Prediction

This project predict customer churn by assessing their to churn. It uses Random forest classifier, K nearest neighbor and Logistic regression for that purpose. I had used backward elimination regression to eliminate the weakly correlated variables by comparing P-value, R<sup>2</sup> and adjusted R<sup>2</sup> values.

## Data Description

Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
The data set includes information about:
* Customers who left within the last month – the column is called Churn
* Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
* Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
* Demographic info about customers – gender, age range, and if they have partners and dependents

[Dataset Source](https://www.kaggle.com/blastchar/telco-customer-churn)


## Technologies Used

### Machine Learning Library:
* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn

### Requirements:
* Python 3.6

## Visualizations
### Model performance
![alt text](https://github.com/krshubham12/Telco-Customer-Churn-Prediction/blob/master/modelperformance.png)

### Correlation between churn and other variables
![alt text](https://github.com/krshubham12/Telco-Customer-Churn-Prediction/blob/master/correlation.png)

### Heatmap
![alt text](https://raw.githubusercontent.com/krshubham12/Telco-Customer-Churn-Prediction/master/heatmap.png)
