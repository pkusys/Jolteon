from sklearn.base import BaseEstimator
from scipy.special import softmax
import numpy as np
from joblib import dump, load
from multiprocessing import Process

class MergedLGBMClassifier(BaseEstimator):
    def __init__(self, model_list):
        assert isinstance(model_list, list)
        
        self.model_list = model_list

    def predict(self, X):
        pred_list = []
        
        for m in self.model_list:
            pred_list.append(m.predict(X))

        # Average the predictions
        averaged_preds = sum(pred_list) / len(pred_list)

        return averaged_preds
    
    def save_model(self, model_path):
        dump(self, model_path)
    
    @staticmethod
    def load_model(model_path):
        return load(model_path)
    
class MyProcess(Process):
    def __init__(self, target, args):
        super().__init__()
        assert callable(target)
        self._result = None
        self._my_function = target
        self._args = args

    def run(self):
        if self._args is None:
            result = self._my_function()
        else:
            result = self._my_function(self._args)
        self._result = result

    @property
    def result(self):
        return self._result