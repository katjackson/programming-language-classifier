from plc_code import *
import numpy as np

test_text = 'This {string} is ([a]) SAMPLE for (testing)\n these &functions.'
test_end = 'The end.'


def test_caps_to_non():
    x = caps_to_non(test_text)
    print(x)
    assert x == 0.19444444444444445


def test_percent_occurence_of_parenthesis():
    x = percent_occurence_of_parenthesis(test_text)
    print(x)
    assert x == 0.06451612903225806


def test_percent_occurence_of_curly():
    x = percent_occurence_of_curly(test_text)
    print(x)
    assert x == 0.03225806451612903


def test_percent_occurence_of_space():
    x = percent_occurence_of_space(test_text)
    print(x)
    assert x == 0.14516129032258066


# I don't know how to test this functon because it has another function in it.
# def test_percent_occurence_of_this_pattern():
#     text = test_text
#     x = percent_occurence_of_this_pattern('&\w')
#     print(x)
#     assert x == feature_fn(test_text)

# def test_feature_fn():
#     reg_ex = '&\w'
#     x = feature_fn(test_text)
#     assert x == 0.016129032258064516


def test_featurizer_transform():
    featurizer = FunctionFeaturizer(
                                    percent_occurence_of_parenthesis,
                                    percent_occurence_of_curly,
                                    percent_occurence_of_space,
                                    )
    featurizer.fit(test_text)
    feature_vector = featurizer.transform(test_text)
    print(type(feature_vector))
    assert type(feature_vector) == np.ndarray
