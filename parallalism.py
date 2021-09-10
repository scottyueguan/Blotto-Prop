from multiprocessing import Pool
from typing import List
import numpy as np

# TODO: enable parallel run
def is_same(pair):
    a, b = pair[0], pair[1]
    if abs(a - b) < 1e-4:
        return True
    else:
        return False

class Worker():
    def __init__(self, n_workers, initializer=None, initargs=None):
        self.pool = Pool(processes=n_workers,
                         initializer=initializer,
                         initargs=initargs)
        self.n_workers = n_workers

    def callback(self, result):
        if result:
            print("Solution found!")
            self.pool.terminate()

    def do_job(self, all_inputs: List):
        splitted_inputs = np.array_split(all_inputs, self.n_workers)
        for inputs in splitted_inputs:
            self.pool.apply_async(is_same,
                                  args=inputs,
                                  callback=self.callback)

        self.pool.close()
        self.pool.join()


if __name__ == "__main__":



    worker = Worker(n_workers=4)
    data = [[np.random.random(1), np.random.random(1)] for _ in range(100000)]

    worker.do_job(data)
