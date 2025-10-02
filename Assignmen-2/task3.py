import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd


data = pd.read_csv(r"C:\AI_With_Python\AI-with-Python\Assignmen-2\weight-height.csv", skiprows=1, names=['x','y'])

height= data['x']
weight= data['y']

height_cm = height * 2.54
weight_kg = weight *0.453592

heigh_mean = np.mean(height_cm)
weight_mean = np.mean(weight_kg)


pyplot.hist(height_cm, bins=10, color='blue',edgecolor='black')

pyplot.title('Graph')
pyplot.legend()
pyplot.grid(True)
pyplot.show()