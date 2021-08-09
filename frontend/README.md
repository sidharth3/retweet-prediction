# Offensive Tweet Detection

This project is part of 50.038 Computational Data Science Module (SUTD).

Our team aims to explore the data science segment of Multi-Class Text Classification, specifically in the sphere of offensive language detection. Using labelled twitter data, we explored various text embedding methods, followed with the implementation of statistical and deep learning models for hierarchical class prediction.

We created a UI using VueJS that mimics the Twitter homepage. After a user types his tweet and clicks the 'Tweet' button, our machine learning model predicts whether the tweet is deemed as offensive or not. If the tweet is not offensive, the results will be shown and he tweet gets pushed onto the homepage. On the other hand, if the tweet is deemed as offensive, there will be an alert button that tells the user that the tweet is offensive and the tweet does not get pushed to the Twitter homepage.

## To run the UI app,

```
npm install
http-server -c-1 -p 8010
```

## To run the Machine Learning Model Backend,

```
Clone code under flask branch
Activate virtual environment using anaconda prompt
Run python NN_Tweet.py
```
