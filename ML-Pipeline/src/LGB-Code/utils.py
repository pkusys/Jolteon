from sklearn.base import BaseEstimator
from scipy.special import softmax
import numpy as np
from joblib import dump, load
from multiprocessing import Process
import boto3
import time

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
    
def get_files(bucket_name, key):
    assert isinstance(key, str)
    
    s3_client = boto3.client('s3')
    
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=key
    )
    
    res = []
    if 'Contents' in response:
        for file in response['Contents']:
            res.append(file['Key'])
    else:
        raise Exception('No files found')
    return res

# MyPool run() will not finish and sticks into the while loop
class MyPool:
    def __init__(self, size, processes):
        assert size > 0
        assert isinstance(size, int)
        assert isinstance(processes, list)
        assert len(processes) > 0
        assert all(isinstance(p, Process) for p in processes)
        
        self.size = size
        self.processes = processes
        self.start = [False] * len(processes)
        self.finish = [False] * len(processes)
        self.quota = self.size
        
    def update(self):
        for i in range(len(self.processes)):
            if self.start[i] and not self.finish[i]:
                if self.processes[i].is_alive():
                    continue
                else:
                    self.finish[i] = True
                    self.quota += 1
        
    def run(self):
        index = 0
        num = len(self.processes)
        t0 = time.time()
        while self.finish[num - 1] is False:
            self.update()
            ct = time.time()
            if ct - t0 > 15:
                print('quota: ', self.quota)
                print('index: ', index)
                print(self.processes[0].is_alive())
                break
            if self.quota > 0 and index < num:
                assert self.start[index] is False
                self.processes[index].start()
                self.start[index] = True
                self.quota -= 1
                index += 1
        for p in self.processes:
            p.join()