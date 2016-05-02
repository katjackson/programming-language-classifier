import numpy as np
import glob
import csv
import re
from sklearn.cross_validation import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.base import TransformerMixin
# from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Used glob to read in program files for training. Defined X and y.
def read_prog_files(loc):
    files = glob.glob(loc, recursive=True)
    texts = []
    for file in files:
        with open(file) as f:
            texts.append(f.read())
    return texts

# file extensions correspond to language names in ext_dict
file_extensions = ['gcc', 'c', 'csharp', 'sbcl', 'clojure', 'java',
                   'javascript', 'ocaml', 'perl', 'hack', 'php', 'python3',
                   'jruby', 'yarv', 'scala', 'racket']
ext_dict = {'jruby': 'ruby', 'csharp': 'c#', 'hack': 'php', 'sbcl':
            'common lisp', 'ocaml': 'ocaml', 'python3': 'python', 'php': 'php',
            'perl': 'perl', 'racket': 'scheme', 'c': 'c', 'javascript':
            'javascript', 'gcc': 'c', 'yarv': 'ruby', 'java': 'java',
            'clojure': 'clojure', 'scala': 'scala'}

X = []
y = []
for ext in file_extensions:
    x_texts = read_prog_files('''/Users/kathrynjackson/Code/homework/programming-language-classifier/benchmarksgame-2014-08-31/benchmarksgame/bench/**/*.{}'''.format(ext))
    X += x_texts
    y += (len(x_texts) * [ext_dict[ext]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    train_size=0.6,
                                                    random_state=890)


"""
I used the following functions and created the FunctionFeaturizer class, a
subclass of TransformerMixin, in an effort to create my own classifier.
"""


def caps_to_non(text):
    cap_letters = re.findall(r'[A-Z]', text)
    non_caps = re.findall(r'[a-z]', text)
    return len(cap_letters) / len(non_caps)


def percent_occurence_of_parenthesis(text):
    pars = re.findall(r'\(|\)', text)
    return len(pars) / len(text)


def percent_occurence_of_curly(text):
    curls = re.findall(r'\{|\}', text)
    return len(curls) / len(text)


def percent_occurence_of_space(text):
    spaces = re.findall(r'\s', text)
    return len(spaces) / len(text)


def percent_occurence_of_this_pattern(reg_ex):

    def feature_fn(text):
        occ = re.findall(r'{}'.format(reg_ex), text)
        return len(occ) / len(text)

    return feature_fn


class FunctionFeaturizer(TransformerMixin):
    def __init__(self, *featurizers):
        self.featurizers = featurizers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fvs = []
        for text in X:
            fv = [f(text) for f in self.featurizers]
            fvs.append(fv)
        return np.array(fvs)

# Instantiate featurizer class
featurizer = FunctionFeaturizer(caps_to_non,
                                percent_occurence_of_parenthesis,
                                percent_occurence_of_curly,
                                percent_occurence_of_space,
                                percent_occurence_of_this_pattern('&\w'),
                                percent_occurence_of_this_pattern('\$\w'),
                                percent_occurence_of_this_pattern(
                                    '[A-Za-z]+[A-Z]'))
# Instantiate count vectorizer
cv = CountVectorizer(token_pattern=r'[a-zA-Z]{2,}|\s|[^\w\d\s]',
                     lowercase=False)
# Combine my featurizer with the count vectorizer
feature_extractors = FeatureUnion([('my featurizer', featurizer), ('cv', cv)])
# Create a pipeline for transforming the data
my_classifier = Pipeline([
                    ('featurizer', feature_extractors),
                    ('classifier', DecisionTreeClassifier(criterion='entropy',
                                                          min_samples_split=1,
                                                          random_state=1067)),
                    ('linsvc', LinearSVC(random_state=13)),
                    ])

my_classifier.fit(X_train, y_train)
my_classifier.named_steps['featurizer'].transform(X_train)
print(my_classifier.score(X_test, y_test))


# the following code reads in the test files from the assignment
testy_X = read_prog_files('/Users/kathrynjackson/Code/homework/assignments-master/week5/polyglot/test/*')
testy_y = []
with open('/Users/kathrynjackson/Code/homework/assignments-master/week5/polyglot/test.csv') as test_targets:
    lines = csv.reader(test_targets)
    for line in lines:
        testy_y.append(line[1])

print(my_classifier.score(testy_X, testy_y))
print(my_classifier.predict(testy_X))

# These metrics prove that I built a crappy classifier
print(classification_report(my_classifier.predict(testy_X), testy_y))
print(confusion_matrix(testy_y, my_classifier.predict(testy_X)))


# This function utilizes my classifier to incorrectly predict the language
def language_guesser(snippet):
    return my_classifier.predict([snippet])
