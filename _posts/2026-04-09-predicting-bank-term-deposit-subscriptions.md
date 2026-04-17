---
layout: post
title: "Predicting Bank Term Deposit Subscriptions with Machine Learning"
date: 2026-04-09
categories: [data-science, machine-learning]
tags: [python, pandas, scikit-learn, classification, banking]
---

Every time a bank calls you trying to sell a product, there is a team behind that campaign hoping you will say yes. But most people do not. In this project, I worked with real data from a Portuguese bank's phone-based marketing campaigns to answer two questions: who is most likely to subscribe to a term deposit, and can we build a model to predict it?

Here is how it went, from first look at the data to final recommendations.

---

## What Is a Term Deposit?

A term deposit is a savings product where a customer locks away a fixed sum of money for a set period in exchange for a guaranteed interest rate. Banks like selling them because they provide a stable source of funds. Customers like them because the returns are predictable.

The bank in this dataset ran campaigns where staff called customers directly and tried to convince them to open one of these accounts. The data records the outcome of each call, along with details about the customer and the broader economic environment at the time.

---

## The Data

The dataset contains **41,188 records**, one per customer contact, with 21 columns covering:

- **Customer details** - age, job type, marital status, education level, existing loans
- **Campaign details** - how many times the customer was called, when they were last contacted, and the outcome of any previous campaign
- **Economic indicators** - the Euribor interest rate, consumer confidence index, and employment figures

The target column is simply called `y`, and it records whether the customer said yes or no.

One thing that jumped out immediately: only about **11% of customers subscribed**. That means the data is heavily skewed. Nine out of ten calls ended in a no. This matters a lot when it comes to building a model, because a model that just guessed "no" every time would be right 89% of the time but completely useless. More on how I handled this later.

---

## Exploring the Data

Before building anything, I spent time getting to know the data. Three questions shaped this phase.

### Who is most likely to subscribe?

Looking at subscription rates by job type, two groups stood out: **students** and **retirees**. Both had significantly higher rates than average.

This makes intuitive sense. Students may be looking to grow small savings they have, while retirees often have lump sums from pensions or property sales and are interested in low-risk, guaranteed returns. These are the people the bank should be prioritising.

### When should you run the campaign?

May had the highest raw number of subscriptions, but that was mainly because it was the busiest month for calls overall. When you look at the proportion of calls that converted, **March, September, October, and December** came out ahead.

The lesson here is that volume and success rate are not the same thing. Running more calls in May does not make May the best month. Scheduling campaigns around the months with better conversion rates would likely improve results without increasing costs.

### Does calling someone more times help?

This was one of the more striking findings. Subscription rates were highest when a customer was contacted **once or twice**. After that, the rate fell off quickly. By the time a customer had been called five or more times, the chance of them saying yes was very low.

This suggests that repeatedly chasing customers who have not responded is a waste of resources. A better approach would be to move on after two attempts and redirect that effort toward new prospects.

---

## Preparing the Data for Modelling

Raw data rarely goes straight into a machine learning model. There are always a few things to clean up and transform first.

**Removing a leaky column.** The dataset included a column called `duration`, which recorded how long each phone call lasted. This is a very strong predictor of whether someone subscribed (longer calls suggest a more engaged customer), but it creates a problem: you only know the call duration after the call has ended. At the point when the bank is deciding who to call, this information does not exist yet. Keeping it in the model would give artificially inflated results that would not hold up in practice. So it was removed.

**Simplifying a messy column.** The `pdays` column recorded how many days had passed since the customer was last contacted in a previous campaign. Customers who had never been contacted were given the value 999. Rather than leaving this as a number, I converted it into a simple yes/no flag: was this customer previously contacted? This is easier for a model to use.

**Encoding categories as numbers.** Machine learning models work with numbers, not words. Columns like job type, education level, and contact method had to be converted into a numerical format. I used a standard technique called one-hot encoding, which creates a separate column for each possible value and marks it with a 1 or 0.

**Splitting the data.** I held back 20% of the data as a test set, keeping it completely separate from training. This gives an honest measure of how well the model performs on data it has never seen. I made sure the 11% subscription rate was preserved in both halves (this is called stratified splitting).

**Scaling the numbers.** Some columns have values in the thousands (like the number of bank employees), while others are small decimals (like the consumer confidence index). For the Logistic Regression model, having everything on a consistent scale helps the algorithm work properly. I applied standard scaling, which centres each column around zero with a standard spread.

---

## Building the Models

I trained two models and compared them.

### Logistic Regression

This is one of the oldest and most interpretable classification algorithms. It draws a straight-line boundary between the two groups (subscribed vs. not subscribed) and estimates the probability of each outcome. It is transparent, fast to train, and easy to explain. Think of it as the sensible baseline.

### Random Forest

This is a more powerful but less transparent approach. It builds hundreds of small decision trees, each trained on a slightly different random sample of the data, and combines their predictions. The ensemble effect tends to produce more accurate results, especially when the relationships in the data are complex and non-linear.

Both models were configured to account for the class imbalance. I used a setting called `class_weight='balanced'`, which tells the model to pay more attention to the minority class (the subscribers) rather than defaulting to always predicting the majority (non-subscribers).

---

## Results

The Random Forest outperformed Logistic Regression on the key metric used here: **ROC-AUC** (pronounced "roc-awk"). This measures how well a model can tell the difference between the two classes across all possible decision thresholds. A score of 1.0 is perfect. A score of 0.5 is no better than flipping a coin.

Both models scored well above 0.5, meaning they genuinely learned useful patterns. The Random Forest scored higher, suggesting the relationship between the features and the outcome has non-linear elements that the simpler model could not fully capture.

---

## What Actually Drives Subscription?

One of the most valuable things a Random Forest can tell you is which features it relied on most heavily. The top predictors were:

- **Euribor 3-month rate** - when interest rates are lower, term deposits are relatively more attractive, and customers are more likely to subscribe
- **Number of employees** - a proxy for broader economic conditions; when the economy is doing well, people may be more confident making financial commitments
- **Consumer confidence index** - a direct measure of how optimistic people feel about the economy
- **Previous campaign outcome** - customers who said yes in a past campaign are much more likely to say yes again

What this tells the bank is that the decision to subscribe is heavily influenced by things outside the bank's direct control, specifically the macroeconomic environment. The best strategy is to run intensive campaigns when conditions are favourable.

---

## What I Would Tell the Bank

Based on everything above, here are five concrete recommendations:

1. **Prioritise students and retirees.** These groups convert at significantly higher rates than average. Targeting them more deliberately would improve overall efficiency.

2. **Run campaigns in March, September, October, and December.** These months show better conversion rates relative to the number of calls made.

3. **Stop calling after two attempts.** The data clearly shows that persistence beyond two calls rarely pays off. Redirect that effort toward new prospects instead.

4. **Give priority to customers who said yes in a previous campaign.** Prior positive engagement is one of the strongest signals in the entire dataset.

5. **Watch the economic indicators.** When the euribor rate is falling and consumer confidence is rising, that is the time to push hardest. Timing campaigns around favourable conditions could meaningfully improve results without increasing the number of calls made.

---

## Wrapping Up

This project is a good example of how data science can go beyond answering "will this customer subscribe?" and start answering "what should we actually do differently?" The model is useful, but the exploratory findings and feature importance analysis are where the actionable insight lives.

If you want to see the full code and visualisations, the notebook is available on [GitHub](#).

---

*Built with Python, pandas, matplotlib, and scikit-learn.*
