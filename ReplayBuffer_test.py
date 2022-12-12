from ReplayBuffer import ReplayBuffer
import numpy as np

a = np.array([1, 1, 1, 1])
b = np.array([2, 2, 2, 2])
c = np.array([3, 3, 3, 3])
r = ReplayBuffer(10, 2, a.shape)


r.store(a, 1, 1, b, True)
r.store(a, 1, 1, b, True)
r.store(b, 1, 1, c, True)
r.store(b, 1, 1, c, True)
r.store(c, 1, 1, a, True)
r.store(c, 1, 1, a, True)


print(r.sample())
