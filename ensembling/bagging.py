import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

class BaggingClassifier:
    """
    Bagging classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__(self, legalLabels, max_iterations, weak_classifier, ratio, num_classifiers):

        self.legalLabels = legalLabels
        self.ratio = ratio
        self.num_classifiers = num_classifiers
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.num_classifiers)]

    def train(self, trainingData, trainingLabels):
        """
        The training loop samples from the data "num_classifiers" time. Size of each sample is
        specified by "ratio". So len(sample)/len(trainingData) should equal ratio. 
        """

        self.features = trainingData[0].keys()
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        for i in range(self.num_classifiers):
            sample_size = int(self.ratio * len(trainingData))
            indices = np.random.choice(range(len(trainingData)), size = sample_size, replace = True)
            sample_train_data = [trainingData[index] for index in indices]
            sample_train_labels = [trainingLabels[index] for index in indices]
            self.classifiers[i].train(sample_train_data, sample_train_labels)

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legalLabels.
        """

        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        guesses = []
        for datum in data:
            result = [self.classifiers[i].classify([datum])[0] for i in range(self.num_classifiers)]
            guess = int(np.sign(np.sum(result)))
            if (guess == 0):
                guess = np.random.choice(self.legalLabels)
            guesses.append(guess)
        return guesses
