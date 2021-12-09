import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.random.seed(0)
sample_per_class = 200


train = pd.read_csv('train.csv') # train
train.head()
print(train.shape)
label = train["label"]
num_class = len(label.value_counts())
train = train.sample(n=num_class * sample_per_class,axis='rows', random_state = 0)

label = train["label"]
print(train.isnull().any().sum())


print(label.value_counts())


train = StandardScaler().fit_transform(train)

tsne = TSNE(n_components = 2, random_state=0)
tsne_res = tsne.fit_transform(train)

plt.figure(figsize=(16,10))

sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = label, palette = sns.hls_palette(10), legend = 'full')

# plt.show()
folder_name = 'figs/'
plt.savefig(folder_name+'tsne_vanilia.png',bbox_inches='tight',  pad_inches = 0)




