Fraud Finder: Identifying Sketchy Transactions
What this does
It is a Python tool that looks at credit card data and tries to spot the difference between a normal purchase and fraud.

The Problem
Fraud is rare. If you have 10,000 transactions, maybe only 20 are fake. Most models get "lazy" and just assume everything is fine because the data is so one-sided. I used a technique called SMOTE to help the computer practice finding those rare cases so it does not miss them in the real world.

How it works
Clean the data: I used Pandas to get everything organized.

Balance the scales: I made sure the model saw enough fraud examples to actually learn what they look like.

Train the AI: I used a Logistic Regression model to do the heavy lifting.

Save for later: I saved the finished model as fraud_model.pkl so it can be reused without retraining.

How to use it
Install the libraries: pip install pandas scikit-learn imbalanced-learn seaborn

Put your data in a file named creditcardinfo.csv

Run main.py and check the results

My takeaway
Data is rarely perfect. This project taught me how to fix unbalanced data so that a model can actually protect people from having their money stolen.
