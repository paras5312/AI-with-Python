import numpy as np
import matplotlib.pyplot as pyplot


x = np.arrange(1, 10)
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

pyplot.figure()
pyplot.scatter(x, y, marker="+", color='violet')
pyplot.title('Scater point')
pyplot.xlabel('x axis')
pyplot.ylabel('y axis')


pyplot.grid(True)
pyplot.show()