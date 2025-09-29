import np as np

A = np.array([[1, 2, 3],  [0, 1, 4],   [5, 6, 0]])

A_inv = np.linalg.inv(A)
print("Inverse of A:    \n  ", A_inv)

I1 = np.dot(A, A_inv)
I2 = np.dot(A_inv, A)

print("A * A_inv:\n\n", np.round(I1, 2))
print("A_inv * A:\n\n\n", np.round(I2, 2))