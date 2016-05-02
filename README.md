## Programming Language classifier
This is an attempt to create a classifier that would classify a snippet of code into one of 13 different languages. Using modules fro scikit-learn, I vectorized the files and tried to classify them using multiple models. I created a custom feature vectorizer to combine with scikit's count vectorizer. However, my classifier is extremely unsuccessful.

My findings are displayed in [this jupyter notebook](https://github.com/katjackson/programming-language-classifier/blob/master/programming_language_classifier.ipynb). The repo also holds a module that holds the notebook code (plc_code.py) and a few tests for the featurizer that do not work as expected (featurizer_tests.py).
