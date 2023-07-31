from multiprocessing import Process
import threading

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

class MyQueue():
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