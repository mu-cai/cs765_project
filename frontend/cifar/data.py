import pandas as pd

data = pd.read_csv("mnist_sample_per_class_50_num_neighbors_3.csv")
for i in range(len(data)):
    for j in data.columns[1:]:
        data[j][i] = data[j][i][1:-1]
data.to_csv("new_mnist_sample_per_class_50_num_neighbors_3.csv")