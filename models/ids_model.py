from sklearn.ensemble import IsolationForest

class IDSModel:
    def __init__(self):
        self.model = IsolationForest(contamination=0.2)

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)