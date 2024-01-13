from multiprocessing import Process, Queue
import threading
import numpy as np
import heapq

class MyProcess(Process):
    def __init__(self, target, args, queue=None):
        super().__init__()
        assert callable(target)
        self._result = None
        self._my_function = target
        self._args = args
        self.queue = queue

    def run(self):
        if self._args is None:
            result = self._my_function()
        else:
            result = self._my_function(self._args)
        if self.queue:
            self.queue.put(result)

class MyThread(threading.Thread):
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

class MyQueue:
    def __init__(self, init_list = None):
        self.queue = [] if init_list is None else init_list

    def push(self, item):
        self.queue.append(item)

    def pop(self):
        assert len(self.queue) > 0
        return self.queue.pop(0)

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return str(self.queue)
    
class Distribution:
    def __init__(self, init_list, init_prob = None):
        assert isinstance(init_list, list)
        assert isinstance(init_prob, list) or init_prob is None
        if len(init_list) == 0:
            raise ValueError('Empty list')
        init_list.sort()
        self.data = np.array(init_list)
        if init_prob is None:
            self.prob = np.ones(len(init_list)) / len(init_list)
        else:
            assert len(init_list) == len(init_prob)
            self.prob = np.array(init_prob)
        
        self.subset = []
        
    def combine(self, dist, com_type = 0):  # com_type: 0 for in_serie, 1 for parallel
        assert isinstance(dist, Distribution)
        assert isinstance(com_type, int)
        
        if com_type == 0:
            new_data = []
            for i in range(len(self.data)):
                for j in range(len(dist.data)):
                    new_data.append( (self.data[i] + dist.data[j], self.prob[i] * dist.prob[j]))
                    
            new_data.sort()
            self.data = np.array([item[0] for item in new_data])
            self.prob = np.array([item[1] for item in new_data])
        elif com_type == 1:
            cum_prob1 = np.cumsum(self.prob)
            cum_prob2 = np.cumsum(dist.prob)
            
            new_data = np.concatenate((self.data, dist.data))
            new_data.sort()
            new_probs = []
            prev = 0
            for item in new_data:
                index = np.sum(self.data <= item) - 1
                prob1 = cum_prob1[index] if index >= 0 else 0
                index = np.sum(dist.data <= item) - 1
                prob2 = cum_prob2[index] if index >= 0 else 0
                new_probs.append(prob1 * prob2 - prev)
                prev = prob1 * prob2
                
            new_data = np.array(new_data)
            new_prob = np.array(new_probs)
            new_prob = new_prob / np.sum(new_prob)
            
            self.data = np.array([])
            self.prob = np.array([])
            
            for i in range(len(new_data)):
                if new_prob[i] > 0:
                    self.data = np.append(self.data, new_data[i])
                    self.prob = np.append(self.prob, new_prob[i])
            
        else:
            raise ValueError('Unknown combine type')
        
        self.reduce_dim()
        
    def probility(self, value):
        val = np.sum(np.array(self.data) <= value)
        if val == 0:
            return 0
        if val == len(self.data):
            return 1
        lower_idx = val - 1
        upper_idx = val
        base_prob = np.sum(self.prob[:lower_idx + 1])
        lower_val = self.data[lower_idx]
        upper_val = self.data[upper_idx]
        
        add_prob = (value - lower_val) / (upper_val - lower_val) * self.prob[upper_idx]
        
        return base_prob + add_prob
    
    def tail_value(self, percentile):
        cum_prob = np.cumsum(self.prob)
        index = None
        for i in range(len(self.prob)):
            if cum_prob[i] >= percentile:
                index = i
        if index is None:
            return self.data[-1]
        if index == 0:
            lower_val = 0
            lower_prob = 0
        else:
            lower_val = self.data[index - 1]
            lower_prob = cum_prob[index - 1]
            
        upper_val = self.data[index]
        upper_prob = cum_prob[index]
        
        add_val = (percentile - lower_prob) / (upper_prob - lower_prob) * (upper_val - lower_val)
            
        return lower_val + add_val
    
    def reduce_dim(self, dim = 100):
        length = len(self.data)
        if length <= dim:
            return
        split_data = np.array_split(self.data, dim)
        split_prob = np.array_split(self.prob, dim)
        
        new_data = []
        new_prob = []
        
        for idx in range(dim):
            probs = split_prob[idx]
            prob = float(np.sum(split_prob[idx]))
            probs = probs / prob
            val = float(np.dot(split_data[idx], probs))
            new_prob.append(prob)
            new_data.append(val)
        
        self.data = np.array(new_data)
        self.prob = np.array(new_prob)
        
    def copy(self):
        new_data = self.data.copy().tolist()
        new_prob = self.prob.copy().tolist()
        return Distribution(new_data, new_prob)
    
    def __str__(self):
        return str(list(zip(self.data, self.prob)))

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        heapq.heappush(self.heap, (-priority, item))

    def pop(self):
        if self.heap:
            priority, item = heapq.heappop(self.heap)
            return item, -priority
        raise IndexError("pop from an empty priority queue")

    def peek(self):
        if self.heap:
            priority, item = self.heap[0]
            return -priority, item
        raise IndexError("peek from an empty priority queue")

    def is_empty(self):
        return len(self.heap) == 0
    
    def __len__(self):
        return len(self.heap)