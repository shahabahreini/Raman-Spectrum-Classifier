import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_ = pd.read_csv('data/WT/RAB2.csv').T
matrix = data_.corr().round(2)
sns.heatmap(matrix)
plt.show()
