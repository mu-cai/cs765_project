import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# import umap
import umap.umap_ as umap
import numpy as np
np.random.seed(0)

sample_per_class = 200 # 10
sns.set(context="paper", style="white")

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

# tsne = TSNE(n_components = 2, random_state=0)
# tsne_res = tsne.fit_transform(train)

reducer = umap.UMAP(random_state=0) # 42
embedding = reducer.fit_transform(train)



plt.figure(figsize=(16,10))

sns.scatterplot(x = embedding[:,0], y = embedding[:,1], hue = label, palette = sns.hls_palette(10), legend = 'full')

# plt.show()
folder_name = 'figs/'
plt.savefig(folder_name+'umap_vanilia.png',bbox_inches='tight',  pad_inches = 0 )




