---
layout: post
title: "Predicting Customer Churn with Machine Learning"
date: 2026-04-09
categories: [machine-learning, data-science]
tags: [python, churn-prediction, scikit-learn, xgboost]
---


Imagine you run a telecom company. A new competitor has just entered the market and is offering attractive deals to new customers. You start worrying: which of your existing customers are most likely to leave? And more importantly, is there anything you can do to keep them?

That is exactly the problem I set out to solve in this project. Using a dataset from a real Iranian telecom company, I built a machine learning model that predicts whether a customer is likely to churn (i.e. cancel their service). Along the way, I also uncovered the key reasons why customers leave in the first place.

Here is how I approached it, step by step.

---

## The Data

The dataset contains information on 3,150 customers collected over a 12-month period. For each customer, we have details like:

- How many calls they made and how long those calls lasted
- How many text messages they sent
- Whether they ever filed a complaint
- How long they have been a subscriber
- Whether they ultimately churned or stayed

The target variable is simple: 1 means the customer left, 0 means they stayed. About 16% of customers in the dataset churned, which means the data is imbalanced. We have far more examples of customers who stayed than customers who left. Handling that imbalance correctly turned out to be one of the most important steps in the project.

---

## Step 1: Exploring the Data

Before building any model, I spent time getting to know the data. This stage is called Exploratory Data Analysis (EDA), and it is where you ask questions and let the data answer them through charts and statistics.

A few things stood out immediately.

**Complaints are a major red flag.** Customers who filed a complaint churned at a dramatically higher rate than those who did not. This was the single strongest signal in the entire dataset.

**Inactive customers almost always leave.** The dataset tracks whether a customer is active or non-active. Unsurprisingly, non-active customers churn at a very high rate.

**Newer customers are more likely to leave.** Customers who had only been subscribed for a short time were much more likely to churn than long-standing customers. Loyalty takes time to build.

**Pay-as-you-go users churn more.** Customers on a contractual plan stayed more often than those paying per use.

I also explored some specific questions about usage patterns, for example which age groups send more SMS messages than they make calls, and whether call duration differs significantly between tariff plans. (It does, with a statistically significant difference confirmed through a Mann-Whitney U test.)

---

## Step 2: Preparing the Data

Raw data is rarely ready to feed directly into a machine learning model. In this project, the main preparation steps were:

**Encoding categorical-style columns.** Two columns, Tariff Plan and Status, used the values 1 and 2 to represent categories (e.g. 1 = active, 2 = non-active). I converted these to 0 and 1, which is the format machine learning models expect.

**Handling the class imbalance.** Because only 16% of customers churned, a lazy model could achieve 84% accuracy just by predicting "no churn" for everyone. That would be useless in practice. To fix this, I used a technique called SMOTE (Synthetic Minority Oversampling Technique), which generates artificial examples of the minority class (churners) so that the model gets a more balanced view of the problem during training.

**Scaling the features.** Different columns have very different numerical ranges. For example, "Seconds of Use" can be in the thousands, while "Complaints" is just 0 or 1. Scaling brings all features onto a comparable range, which helps certain models perform better.

---

## Step 3: Engineering New Features

Sometimes the most useful signals are not directly in the raw data but can be calculated from it. I created four new features:

- **Average Call Duration:** Total seconds of calls divided by total number of calls. This captures how long a typical call is, rather than just the volume.
- **SMS Ratio:** The share of all communications that are text messages. A high ratio might indicate a customer who prefers messaging over calling.
- **Failure Rate:** Call failures divided by total calls. A customer experiencing frequent failures is likely frustrated.
- **Tenure Group:** Subscription length grouped into new (up to 12 months), mid-term (12 to 36 months), and loyal (over 36 months).

---

## Step 4: Training the Models

With the data prepared, I trained seven different machine learning models and compared their performance:

| Model | What it does |
| --- | --- |
| Logistic Regression (L1) | A linear model that also performs automatic feature selection |
| Logistic Regression (L2) | A standard linear baseline |
| Decision Tree | A set of rules that splits customers into groups |
| K-Nearest Neighbours | Predicts based on the most similar past customers |
| Random Forest | Hundreds of decision trees working together |
| Gradient Boosting | Trees that learn from each other's mistakes, one at a time |
| XGBoost | A highly optimised version of gradient boosting |

Training multiple models and comparing them is a best practice. No single algorithm is always the best, and it is worth seeing how they stack up against each other on your specific problem.

To evaluate each model fairly, I held out 20% of the data as a test set that the models never saw during training. Performance was measured using several metrics:

- **Accuracy:** The percentage of predictions that were correct overall.
- **Precision:** Of the customers predicted to churn, how many actually did?
- **Recall:** Of the customers who actually churned, how many did the model catch?
- **F1 Score:** A single number that balances precision and recall.
- **ROC-AUC:** A measure of how well the model distinguishes between churners and non-churners across all possible thresholds.

For churn prediction, recall matters a lot. Missing a customer who is about to leave is costly. A false alarm (predicting churn for someone who would have stayed) is annoying but cheaper to act on.

---

## Step 5: Tuning the Decision Tree

The tutorial approach from DataCamp emphasises interpretability as well as performance. Decision trees are useful because you can actually read the rules they learn. However, trees that grow too deep tend to overfit, meaning they perform well on training data but poorly on new data.

I used a technique called GridSearchCV to systematically test different combinations of tree settings (depth, minimum samples per leaf, and the splitting criterion). This found the optimal configuration automatically, balancing model complexity against generalisation.

---

## Step 6: Understanding What Drives Churn

Predicting churn is only half the job. The business also needs to understand *why* customers are leaving so that it can take action.

**Logistic Regression with L1 regularisation** is particularly useful here. The L1 penalty forces the model to zero out features that are not contributing useful information, leaving only the true drivers. The remaining coefficients can be read directly: a positive coefficient increases churn risk, a negative one decreases it.

**Tree-based models** (like Random Forest and XGBoost) provide feature importance scores, ranking each variable by how much it contributed to the model's predictions.

Both approaches pointed to the same conclusion: complaints, account status, subscription length, call failure rate, and tariff plan are the most important factors.

---

## Key Findings

| Driver | What it tells us |
| --- | --- |
| **Complaints** | The strongest signal. A customer who has complained is far more likely to leave. |
| **Status** | Non-active accounts almost always churn. This is a near-perfect leading indicator. |
| **Subscription Length** | The longer a customer has stayed, the less likely they are to leave. |
| **Call Failure Rate** | Frequent call failures damage the customer experience and drive churn. |
| **Age Group** | Younger customers churn more. Older customers tend to be more loyal. |
| **Tariff Plan** | Pay-as-you-go customers churn more than those on contracts. |

---

## Business Recommendations

Armed with these findings, the telecom company can take targeted action:

1. **Act on complaints immediately.** Set up an automated alert so that any customer who files a complaint is contacted within 24 hours with a resolution offer or retention incentive.

2. **Invest in new customer onboarding.** The first 12 months are the highest-risk period. Welcome calls, check-in emails, and loyalty rewards during this window could significantly reduce early churn.

3. **Fix the network quality issues.** High call failure rates are frustrating customers. Identifying the areas or times with the most failures, and prioritising repairs there, would directly reduce one of the top churn drivers.

4. **Move pay-as-you-go customers onto contracts.** Customers on contracts churn far less. Offering a compelling reason to switch (a discount, extra data, or a free trial period) could lock in more customers for longer.

5. **Target younger customers specifically.** Age Group 1 customers churn the most. Campaigns tailored to their preferences (such as SMS-heavy plans or competitive pricing) could help retain this segment.

6. **Monitor for non-active status.** Non-active customers are on the verge of leaving. Triggering an outreach campaign as soon as a customer goes non-active could intercept churn before it happens.

---

## Reflections

This project reinforced something I find true in most data science work: the most valuable insights are often not from the most complex model. Understanding why customers leave, and being able to communicate that clearly to a business audience, is just as important as the prediction accuracy itself.

The full code and notebook are available on [GitHub](https://github.com).

---

*Built with Python, scikit-learn, XGBoost, imbalanced-learn, pandas, matplotlib, and seaborn.*
