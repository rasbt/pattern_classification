Sebastian Raschka, 2015

# Embedding a machine learning algorithm in a web application

This is the source code of a very simple web application that attempts to detect the sentiment of a movie review (positive or negative). 

**Web app URL:** [http://raschkas.pythonanywhere.com](#http://raschkas.pythonanywhere.com)

The underlying model is a logistic regression classifier trained on the [50,000 movie review dataset](http://ai.stanford.edu/%7Eamaas/data/sentiment/) from IMDb via stochastic gradient descent. The word vectors (1-grams) are created via scikit-learn's [HashingVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html), stop words are removed, and the words are transformed into their root form using the Porter stemmer algorithm implemented in the [NLTK](http://www.nltk.org) library. 
The code for model training can be found in the IPython notebook [web_app.ipynb](./web_app.ipynb).

**Don't worry, I will surely post a blog article or IPython notebook to explain more about this topic ;)**
