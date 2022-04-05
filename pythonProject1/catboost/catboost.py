import numpy as np
from utils.loss_functions import CrossEntropy
import catboost as cb
import progressbar
from utils.misc import bar_widgets

class catboost_classifier():
    """
    This is a catboost classifier
    """

    def __init__(self, iterations=600, depth=9, learning_rate=0.3, l2_leaf_reg=4,):
        self.iterations = iterations  # Numbers of iterations in model
        self.depth = depth  # Number of depth in model
        self.learning_rate = learning_rate  # Step size for weight update
        self.l2_leaf_reg = l2_leaf_reg  # The l2 number to build trees

        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        # Log loss for classification
        self.loss = CrossEntropy()
        self.cb = []
        for _ in range(n_estimators):
            cb = cb.CatBoostClassifier(iterations=self.iterations,
                    depth=self.depth,
                    learning_rate=self.learning_rate,
                    feature_fraction=self.feature_fraction,
                    l2_leaf_reg=self.l2_leaf_reg,
                    loss=self.loss)

        self.cb.append(cb)

    def fit(self, X, y):
        # y = to_categorical(y)
        m = X.shape[0]
        y = np.reshape(y, (m, -1))
        y_pred = np.zeros(np.shape(y))
        for i in self.bar(range(self.n_estimators)):
            gbm = self.cb[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            cb.fit(X, y_and_pred)
            update_pred = cb.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            y_pred += update_pred

    def predict(self, X):
        y_pred = None
        m = X.shape[0]
        # Make predictions
        for cb in self.gbm:
            # Estimate gradient and update prediction
            update_pred = cb.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred += update_pred

        return y_pred

