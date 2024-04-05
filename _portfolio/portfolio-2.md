---
title: "Sentiment Analysis Using PyTorch Project"
excerpt: |
  This was an open-ended project assigned as part of my Master of Science (Human Language Technology) program, in the class Advanced Statistical Natural Language Processing. It had the goal of experimenting with and learning about neural networks. The task I chose was to investigate doing a sentiment analysis on a real-world dataset that I found online. I chose this task because sentiment analysis is an important NLP task, since it can help companies understand what their customers are saying about them, and where they can make improvements. 

  The basic goal of the project was that if the model reads in some text, it will be able to predict the negative, neutral, or positive sentiment. While this is focused on a particular dataset I found on Yelp, the basic architecture I developed could be used on other datasets as well (with additional testing and adjustments to the model based on different styles of texts).

  Please see the Sentiment Analysis Using PyTorch page for detailed information about this project.

collection: portfolio
---

_Main Python toolkits used: pandas, numpy, torch, sklearn, nltk_

### Project Overview
For this project, I used Python to develop a Natural Language Processing (NLP) system aimed at sentiment analysis, specifically targeting Yelp reviews. The Yelp reviews I used for the project are available publicly at this location: [Yelp dataset](https://www.yelp.com/dataset/download)

The code is available on my GitHub, located here: [GitHub respository](https://github.com/kelynnski/sentiment-analysis)

The primary goal of this project was to classify Yelp reviews into three sentiment categories: negative, neutral, and positive. This objective aligns with a common use case in NLP, which is understanding user sentiment from text data, which is valuable for both businesses and researchers. In this case it’s a common business need—understanding what their customers think about their product or service. I believe this is an important part of business, because in order to improve (and therefore gain more revenue), this feedback is crucial and can lead to data-driven decisions about how to implement changes or which changes to implement.
My project was designed to be flexible in handling large datasets, applying NLP techniques to preprocess text, and employing machine learning algorithms for classification, resulting in a system that would be beneficial for any business that collects user reviews.

### Approach

The Yelp review dataset consisted of raw JSON files. It included information such as a customer ID, the text review, and a star rating between 1-5 for restaurant reviews (among some other data points I did not leverage for this project). You can find the link to Yelp's dataset here: 'https://www.yelp.com/dataset/download'. The data was structured so that it included a star rating between 1-5 stars, and I converted these star ratings into a different system, where the 1-2 stars were changed to negative (class 0), 3 stars were neutral (class 1), and 4-5 stars were considered positive (class 2). This step was crucial for transforming subjective sentiment into a quantifiable format for machine learning models.

Since the original dataset was so large, I implemented a chunking method to handle and process the data without overwhelming my machine. I included an easy way to adjust this, so on a machine that could handle more, the chunk sizes could be set to a larger number, and the amount of data to train on could be adjusted. I personally only used 20,000 rows of data, 4,000 of which I set aside to be the fake test data set. Of the random 16,000 rows of data I extracted, 11,240 were positive reviews, 2,962 were negative, and only 1,799 were neutral. This meant I was working with an imbalanced dataset, with way more positive reviews. This was partially due to only including 3-star ratings as neutral while including 1-2 as negative and 4-5 as positive, but even without this simplification, the positive reviews were still a much higher count than neutral/negative. This imbalance became a challenge I faced in this project.
Another challenge that I noticed was that due to the nature of a ton of different people writing reviews, the writing styles were all very different and included many typos and strange formatting. The data was not clean in this way; however, at least the dataset was clean in that there were no null values in the text or the labels provided.

After getting the dataset to a size that I wanted to work with, it was time to clean up and preprocess the review texts. I preprocessed the text by removing stopwords, normalizing text to lowercase, and applying a custom function to extract features. These functions utilized the nltk library's Sentiment Intensity Analyzer (which is a pre-built sentiment analysis library) to calculate compound sentiment scores, adding a layer of sentiment-based feature engineering to the project. I tried some other stylometric features but they did not give a large boost to the validation and hold out accuracy and other metrics like F1-score.

For vectorization, I used a combination of character and word n-gram features through sklearn’s TfidfVectorizer, which were later unified using sklearn’s FeatureUnion. I did experiment using only character n-grams and only word n-grams, but my tests shows the union of both was promising, and resulted in better outcomes in comparison to a bag-of-words approach that I also experimented with in the early stages of the project.

For the actual neural network and training portion of the project, I implemented PyTorch, a popular deep learning library. I developed a sentiment classifier class, which consisted of a FeedForward Neural Network with fully connected layers, activation functions, and a final output layer designed to classify the input into one of the three sentiment categories. I experimented with elements such as the hyperparameters, the number of layers, and the class weights. Ultimately, I went with two layers and the ReLU function, and did not end up adding weights to the underrepresented classes, since it did not seem to improve the performance and F1-scores actually went down.

I was able to keep track of my model's performance by splitting the training set into a main train set, a validation set, and a hold-out set. I added an early stop check during the evaluation, so that if the loss did not decrease, it would break out of the training loop early to prevent overfitting.

### Results

Overall my model ended up performing well, with my classification report showing an accuracy of roughly 92.3% for the validation set, and 91% on the hold-out set. The F1-scores were also pretty high. For the validation set, class 0 was .9, class 1 was .73, and class 2 .96. For the hold-out set, class 0 was .89, class 1 .67 (the lowest), and class 2 .95. This meant the positive class (2) was the highest, and class 1 (neutral) struggled the most. This makes sense as it was the most underrepresented class.

When reviewing where my model struggled with the most, it made the most errors on the true neutral class. On the validation set, it misclassified a total of 65 items, mainly over-predicting positive (46 were classified as positive and 19 as negative). (The total number of misclassifications was 160). On the hold-out set, it also showed the most misclassifications in the true neutral set. It misclassified 172 items as positive when they should have been neutral, with 55 negative (the total misclassifications was 504).
