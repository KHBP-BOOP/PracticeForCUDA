import torch
import numpy as np


# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# print(torch.stack((X, Y), dim=0))

# print(torch.stack((X, Y), dim=1))

# print(torch.stack((X, Y), dim=2))


# a = torch.randn(3, 4)
# unsq_a = a.unsqueeze(1)

# print(unsq_a.size())


X = torch.arange(24).reshape(2,3,4)
print(X.is_contiguous())
print(X)

Y = X.permute(0, 2, 1)
print(Y.is_contiguous())
print(Y)


