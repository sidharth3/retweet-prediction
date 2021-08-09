# 50.021 Artificial Intelligence COVID-19 Retweet Prediction

## Introduction

To better understand the information spreading process during the COVID-19 pandemic, the project chose to focus on Twitter, a social media platform where users can follow others and exchange knowledge via short text posts known as tweets. Specifically, the retweeting function on Twitter was explored. Since retweeting serves as a means to widen the spread of original content, understanding retweet behaviours is useful in understanding the information spreading process. Practical applications include fake news spreading and tracking, health promotion and mass emergency management.

Given the TweetsCOV19 dataset, the group aims to predict the number of times each tweet will be retweeted.

## Dataset

We used the TweetsCOV19 dataset that can be found in https://data.gesis.org/tweetscov19/.

## Architecture

The model adopted is a Deep Multi-Layer Perceptron model. Fully connected linear layers were chosen to tackle the regression problem.

Sparse and dense features are concatenated as input into the model. The model comprises three hidden blocks, followed by a linear prediction layer with a sigmoid activation function. A sigmoid activation function was chosen to predict the scaled regression value.

Each hidden block consists of a linear layer, a batch normalisation layer, ReLU activation function and lastly a dropout layer. The number of neurons in the linear layers of the first, second and third hidden block are 4096, 1024 and 128 respectively.

## Environment Set Up

## Graphical User Interface (GUI) Demonstration

To run the GUI, one has to run both the frontend as well as the backend that does the model prediction for the tweet. Run the following code in the terminal shell:

```
cd frontend
http-server -c-1 -p 8010
```

In another terminal shell, run the following:

```
cd backend
python app.py
```

Note: All dependencies must be installed via pip in order to run the backend code.

## Group Members

Noorbakht Khan (1003827)
<br/>
Sidharth Praveen (1003647)
<br/>
Suhas Sahu (1003370)
<br/>
Tiffany Goh (1003674)
<br/>
