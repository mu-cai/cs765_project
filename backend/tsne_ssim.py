from math import sqrt
from numpy.core.fromnumeric import size
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
np.random.seed(0)
import lpips
import torch
use_cuda = torch.cuda.is_available()


loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
if use_cuda:
    loss_fn_alex = loss_fn_alex.cuda()


index1 = 1
index2 = 1
scale = 255
sample_per_class = 1 # 200 #  200
num_neighbors = 10




search_index = 2
min_index_num = 2



def ssim(img_org, img1, scale):
    return compare_ssim(img_org/scale, img1/scale,multichannel=True)


def ssim_index(train, index1, index2, shape):
    img1 =  train.iloc[index1].values[1:].reshape(shape, shape)
    img2 = train.iloc[index2].values[1:].reshape(shape, shape)
    # plt.imshow(img1)
    # plt.colorbar()
    # plt.show()
    # print('ssim', ssim(img2, img1, scale))
    return ssim(img2, img1, scale)

def lpips_index(train, index1, index2, shape):
    img1 =  train.iloc[index1].values[1:].reshape(shape, shape)
    img2 = train.iloc[index2].values[1:].reshape(shape, shape)
    img1 =(torch.from_numpy(img1)).unsqueeze(0).unsqueeze(0)
    img2 =(torch.from_numpy(img2)).unsqueeze(0).unsqueeze(0)


    # plt.imshow(img1)
    # plt.colorbar()
    # plt.show()
    # print('ssim', ssim(img2, img1, scale))
    if use_cuda:
        img1 = img1.cuda()
        img2 = img2.cuda()
    return loss_fn_alex(img0, img1)


def return_img(train,search_index, shape, show = False):
    img1 =  train.iloc[search_index].values[1:].reshape(shape, shape) 
    if show:
        plt.imshow(img1)
        plt.colorbar()
        plt.show()
    return img1


train = pd.read_csv('train.csv') # train
train.head()
print(train.shape)
label = train["label"]
num_class = len(label.value_counts())
train = train.sample(n=num_class * sample_per_class,axis='rows', random_state = 0)

label = train["label"]
print(train.isnull().any().sum())


print(label.value_counts())

shape = int(sqrt(train.iloc[index1].values[1:].shape[0] )  )
print("train[1,:].shape = ", train.iloc[index1].values[1:].reshape(shape, shape).shape)
img1 =  train.iloc[index1].values[1:].reshape(shape, shape)
img2 = train.iloc[index2].values[1:].reshape(shape, shape)
# plt.imshow(img1)
# plt.colorbar()
# plt.show()


############## LPIPS ##############
lpips_value = ssim_index(train, index1, index2, shape)
print('lpips_value:', lpips_value)


############## SSIM ##############
# ssim_value = ssim_index(train, index1, index2, shape)
# print('ssim', ssim_value)

train_df = train
train = StandardScaler().fit_transform(train)

tsne = TSNE(n_components = 2, random_state=0)
tsne_res = tsne.fit_transform(train)



target_embed = tsne_res [search_index]



# plt.figure(figsize=(16,10))
# sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = label, palette = sns.hls_palette(10), legend = 'full')
# plt.plot(tsne_res[search_index,0], tsne_res[search_index,1],  color='r',markerfacecolor='red',marker='o',markersize=12) # , size = 2
# # plt.show()
# folder_name = 'figs/'
# plt.savefig(folder_name+'tsne_vanilia_specific.png',bbox_inches='tight',  pad_inches = 0)




mse_list = (tsne_res [:, 0] - target_embed [0] ) ** 2  + (tsne_res [:, 1] - target_embed [1] ) ** 2 
# print("mse_list = ", mse_list)

for i in range(num_neighbors):
    min_index = np.argsort(mse_list)[i]
    img = return_img(train_df,min_index, shape, show = False)
    if i == 0:
        img_whole = img
    else:
        img_whole = np.concatenate((img_whole, img), axis=1)
plt.imshow(img_whole)
# plt.colorbar()
# plt.show()
plt.savefig('figs/mse_figures_specific.png',bbox_inches='tight',  pad_inches = 0)


 
min_index = np.argsort(mse_list)[min_index_num]
print("min_index:", min_index)

img = return_img(train_df,min_index, shape, show = False)

img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
if use_cuda:
    img0 = img0.cuda()
    img1 = img1.cuda()
d = loss_fn_alex(img0, img1)









