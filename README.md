# BusinessOnBot-Part-2

# QUESTION
  Perform sentiment analysis on the given dataset. The tweets have been pulled from Twitter and manual tagging has been done. Do the pre - processing and split the dataset into TRAIN and TEST datasets. Create a model which classifies the given data into one of the given labels. And also provide the approach followed step by step.

# SOLUTION
  Sentiment analysis uses machine learning and natural language processing (NLP) to identify whether a text is negative or positive.
  
  Step 1: Feature Extraction
Before the model can classify text, the text needs to be prepared so it can be read by a computer. Tokenization, lemmatization and stopword removal can be part of this process, similarly to rule-based approaches.In addition, text is transformed into numbers using a process called vectorization. These numeric representations are known as “features”. A common way to do this is to use the bag of words or bag-of-ngrams methods. These vectorize text according to the number of times words appear.

Recently deep learning has introduced new ways of performing text vectorization. One example is the word2vec algorithm that uses a neural network model. The neural network can be taught to learn word associations from large quantities of text. Word2vec represents each distinct word as a vector, or a list of numbers. The advantage of this approach is that words with similar meanings are given similar numeric representations. This can help to improve the accuracy of sentiment analysis.

Step 2: Training & Prediction
In the next stage, the algorithm is fed a sentiment-labelled training set. The model then learns to associate input data with the most appropriate corresponding label. For example, this input data would include pairs of features (or numeric representations of text) and their corresponding positive, negative or neutral label. The training data can be either created manually or generated from reviews themselves.

Step 3: Predictions
The final stage is where ML sentiment analysis has the greatest advantage over rule-based approaches. New text is fed into the model. The model then predicts labels (also called classes or tags) for this unseen data using the model learned from the training data. The data can thus be labelled as positive, negative or neutral in sentiment. This eliminates the need for a pre-defined lexicon used in rule-based sentiment analysis.

Classification algorithms
  Classification algorithms are used to predict the sentiment of a particular text. As detailed in the vgsteps above, they are trained using pre-labelled training data. Classification models commonly use Naive Bayes, Logistic Regression, Support Vector Machines, Linear Regression .
  
 # USAGE OF MODELS
  Naive Bayes: this type of classification is based on Bayes’ Theorem. These are probabilistic algorithms meaning they calculate the probability of a label for a particular text. The text is then labelled with the highest probability label. “Naive” refers to the fundamental assumption that each feature is independent. Individual words make an independent and equal contribution to the overall outcome. This assumption can help this algorithm work well even where there is limited or mislabelled data.
  
  ACCURACY :61.98

Logistic Regression: a classification algorithm that predicts a binary outcome based on independent variables. It uses the sigmoid function which outputs a probability between 0 and 1. Words and phrases can be either classified as positive or negative. For example, “super slow processing speed” would be classified as 0 or negative.

  ACCURACY:62.81

Support Vector Machines: a model that plots labelled data as points in a multi-dimensional space. The hyperplane or decision boundary is a line which divides the data points. In the example below, anything to the left of the hyperplane would be classified as negative. And everything to the right would be classified as positive. The best hyperplane is one where the distance to the nearest data point of each tag is the largest. Support vectors are those data points which are closer to the hyperplane. They influence its position and orientation. These are the points which help to build the support vector machine.
  
  ACCURACY:61.64
  
 Random forest :The  random forest  algorithm  can be  used  for both regression and classification tasks. This study conducts a sentimental analysis with data sources from Twitter using the  Random Forest algorithm approach,  we  will  measure  the  evaluation  results  of  the algorithm  we  use  in  this  study. 
 
  ACCURACY:64.32

# CONCLUSION
  The best fit model is Random Forest with an accuracy of 64.32%. The tweets are passed as input to the model and classified as 'Positive' or 'Negative'.
