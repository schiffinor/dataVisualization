"""naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
YOUR NAME HERE
CS 251: Data Analysis and Visualization
Spring 2024
"""
import numpy as np

from classifier import Classifier


class NaiveBayes(Classifier):
    """Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any number of classes)"""

    def __init__(self, num_classes):
        """
        Naive Bayes constructor

        - Call superclass constructor
        - Add placeholder instance variables the class prior probabilities and class likelihoods (assigned to None).
        You may store the priors and likelihoods themselves or the logs of them. Be sure to use variable names that make
        clear your choice of which version you are maintaining.
        """
        super().__init__(num_classes)
        self.priors = None
        self.likelihoods = None

    def get_priors(self):
        """Returns the class priors (or log of class priors if storing that)"""
        return self.priors

    def get_likelihoods(self):
        """Returns the class likelihoods (or log of class likelihoods if storing that)"""
        return self.likelihoods

    def train(self, data, y):
        """Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.


        - Compute the class priors and class likelihoods (i.e. your instance variables) that are needed for
        Bayes Rule. See equations in notebook.
        """
        num_samps, num_features = data.shape
        N = y.shape[0]
        priors = np.zeros((self.num_classes,))
        likelihood = np.zeros((self.num_classes, num_features))

        for c in range(self.num_classes):
            priors[c] = np.count_nonzero(y == c) / N
            index_class = np.where(y == c)[0]
            data_class = data[index_class, :]
            total_word_count = np.sum(data_class)
            word_count = np.sum(data_class, axis=0)
            likelihood[c] = (word_count + 1) / (total_word_count + num_features)

        self.priors = np.log(priors)
        self.likelihoods = np.log(likelihood)


    def predict(self, data):
        """Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: this can also be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        """
        data = np.copy(data).astype(int)
        y_pred = np.zeros((data.shape[0],))
        for i in range(data.shape[0]):
            log_posterior = np.zeros((self.num_classes,))
            for c in range(self.num_classes):
                log_posterior[c] = self.priors[c] + np.sum(self.likelihoods[c] * data[i])
            y_pred[i] = np.argmax(log_posterior)
        return y_pred.astype(int)
