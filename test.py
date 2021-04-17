# class Test(object):
#     def __init__(self):
#         # 1
#         self.agents = []

# test = Test()
# test.agents = [1]

import numpy as np

reward = [100, -120, -10]

print(np.any(np.array(reward) > 0))