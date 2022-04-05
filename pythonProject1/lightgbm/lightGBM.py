import numpy as np
from utils.loss_functions import CrossEntropy
import lightgbm as lgb
import progressbar
from utils.misc import bar_widgets

class lightGBM():
    """
    This is a lightGBM classifier
    """

    def __init__(self, n_estimators=200,n_leaves=30, learning_rate=0.01, feature_fraction=0.9,
                 bagging_fraction=0.8, bagging_freq=5, class_number=3):
        self.n_estimators = n_estimators # Numbers of estimators in model
        self.n_leaves = n_leaves  # Number of leaves in model
        self.learning_rate = learning_rate  # Step size for weight update
        self.feature_fraction = feature_fraction  # The number of features to build trees
        self.bagging_fraction = bagging_fraction  # The number of samples to start sampling
        self.bagging_freq = bagging_freq  # After freq iterations, resampling
        self.class_number= class_number # Number of classes

        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        # Log loss for classification
        self.loss = CrossEntropy()
        self.gbm = []
        for _ in range(n_estimators):
            gbm = lgb.train(n_estimators=self.n_estimators,
                    n_leaves=self.n_leaves,
                    learning_rate=self.learning_rate,
                    feature_fraction=self.feature_fraction,
                    bagging_fraction=self.bagging_fraction,
                    bagging_freq=self.bagging_freq,
                    loss=self.loss)

        self.gbm.append(gbm)

    def fit(self, X, y):
        # y = to_categorical(y)
        m = X.shape[0]
        y = np.reshape(y, (m, -1))
        y_pred = np.zeros(np.shape(y))
        for i in self.bar(range(self.n_estimators)):
            gbm = self.gbm[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            gbm.fit(X, y_and_pred)
            update_pred = gbm.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            y_pred += update_pred

    def predict(self, X):
        y_pred = None
        m = X.shape[0]
        # Make predictions
        for gbm in self.gbm:
            # Estimate gradient and update prediction
            update_pred = gbm.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred += update_pred

        return y_pred

