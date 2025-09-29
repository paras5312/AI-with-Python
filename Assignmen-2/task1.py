import numpy as np
import matplotlib.pyplot as pyplot


x = np.linspace(-5, 5, 100)

y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x + 3

pyplot.figure()

pyplot.plot(x, y1, color="blue", linstyle='-', label="y1")
pyplot.plot(x, y2, color="red", linestyle='--', label="y2")
pyplot.plot(x, y3, color="green", linestyle='---', label="y3")


pyplot.title("Comparison of three linear functions")
pyplot.xlabel('x-axis')
pyplot.ylabel('y-axis')

pyplot.legend()
pyplot.grid(True)
pyplot.show()

