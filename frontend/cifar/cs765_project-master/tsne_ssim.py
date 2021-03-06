from math import sqrt
from numpy.core.fromnumeric import size
import pandas as pd
from skimage._shared.utils import check_nD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import torch.nn.functional as F
np.random.seed(0)
import lpips
from PIL import Image
import torch
import csv, os
use_cuda = torch.cuda.is_available()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--high_score', default='SSIM', type=str, help="")
parser.add_argument('--low_score', default='TSNE', type=str, help="")
parser.add_argument('--dataset', default='mnist', type=str, help="")
parser.add_argument('--sample_per_class', default=10, type=int, help="")
parser.add_argument('--num_neighbors', default=3, type=int, help="")
parser.add_argument('--n_components', default=2, type=int, help="")
args = parser.parse_args()



dataset= args.dataset # 'mnist' # mnist cifar
high_score = args.high_score #  'SSIM'
low_score = args.low_score #  'TSNE' # TSNE UMAP

if high_score == 'LPIPS':
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    if use_cuda:
        loss_fn_alex = loss_fn_alex.cuda()

index1 = 1
index2 = 2
scale = 255
sample_per_class = args.sample_per_class # 10 # 200 #  200
deep_input_size = 32
num_neighbors = args.num_neighbors # 3
show_image = False #  True # False
save_image_to_dir = not show_image # True
search_index = 2
min_index_num = 2

if dataset=='mnist':
    dataset_name = 'train.csv'
    channel_num = 1
elif dataset=='cifar':
    dataset_name= 'train_cifar.csv'
    channel_num = 3
train = pd.read_csv(dataset_name) # train
train.head()
# print(train.shape[0])
label = train["label"]
num_class = len(label.value_counts())
whole_num_sample = num_class * sample_per_class 
run_file_num = whole_num_sample if not show_image else 1
shape = int(sqrt(train.iloc[index1].values[1:].shape[0] /channel_num )  )
print('Image_width: shape')
base_name = f"{dataset}_{high_score}_{low_score}_sample_{whole_num_sample}_num_neighbors_{num_neighbors}"
filename = base_name+ ".csv"
home_dir =  os.environ['HOME']
print(home_dir)
save_img_path = os.path.join(home_dir, f'public/html-s/cs765/{base_name}')
if not os.path.exists(save_img_path) and not show_image:
    os.mkdir(save_img_path)
    # os.makedirs(save_img_path, exist_ok=True)



save_similarity_score = True
if save_similarity_score:
    sim_high = np.zeros((whole_num_sample, whole_num_sample))
    sim_high_filename = base_name+ f"_{high_score}_similarity.npy"
    sim_low = np.zeros((whole_num_sample, whole_num_sample))
    sim_low_filename = base_name+ f"_{low_score}_similarity.npy"



def reshape_img(train,search_index):
    if channel_num==3:
        img1 =  train.iloc[search_index].values[1:].reshape(channel_num, shape, shape ) 
        img1 = np.transpose(img1, (1, 2, 0))
        img1 = img1.astype(np.uint8)
    elif channel_num ==1:
        img1 =  train.iloc[search_index].values[1:].reshape( shape, shape ) 
    return img1


def ssim(img_org, img1, scale):
    return compare_ssim(img_org/scale, img1/scale,multichannel=True)


def ssim_index(train, index1, index2, shape):
    img1 = reshape_img(train,index1)
    img2 = reshape_img(train,index2)
    # plt.imshow(img1)
    # plt.colorbar()
    # plt.show()
    # print('ssim', ssim(img2, img1, scale))
    return ssim(img2, img1, scale)

def lpips_index(train, index1, index2, shape):
    img1 =  train.iloc[index1].values[1:].reshape(shape, shape)/scale
    img2 =  train.iloc[index2].values[1:].reshape(shape, shape)/scale
    # img2 = np.expand_dims(train.iloc[index2].values[1:].reshape(shape, shape),axis=0)
    

    # img1 = Image.fromarray(img2) # , 'RGB'
    img1 =(torch.from_numpy(img1)).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)
    img2 =(torch.from_numpy(img2)).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)

    img1 = F.interpolate(img1, (deep_input_size, deep_input_size))
    img2 = F.interpolate(img2, (deep_input_size, deep_input_size))

    # print(img1.shape, img2.shape)

    # plt.imshow(img1)
    # plt.colorbar()
    # plt.show()
    # print('ssim', ssim(img2, img1, scale))
    if use_cuda:
        img1 = img1.cuda()
        img2 = img2.cuda()
    return loss_fn_alex(img1.long(), img2.long()).item()


def return_img(train,search_index, shape, show = False, save = False):
    img1 =  reshape_img(train,search_index)
    if show:
        plt.imshow(img1)
        plt.colorbar()
        plt.show()
    if save:
        plt.imsave(os.path.join( save_img_path,  f'{search_index}.jpg'), img1)
    return img1

def min_distance(train, index, metric = 'LPIPS', mse_list =None, tsne_res = None):
    if metric == 'LPIPS' or metric == 'SSIM':
        mse_list = []
        for i in range(whole_num_sample):
            if metric == 'LPIPS':
                distance_tmp = lpips_index(train, index, i, shape)
            elif metric == 'SSIM':
                distance_tmp = ssim_index(train, index, i, shape)
            mse_list.append( distance_tmp)
    elif metric == 'general':
        target_embed = tsne_res [index]
        mse_list = (tsne_res [:, 0] - target_embed [0] ) ** 2  + (tsne_res [:, 1] - target_embed [1] ) ** 2 
    else:
        mse_list = mse_list
    min_dis_index_list = []
    min_dis_value_list = []

    for i in range(num_neighbors):
        if metric == 'SSIM':
            comp_mse_list = [x * -1 for x in mse_list ]
        else:
            comp_mse_list = mse_list
        index = np.argsort(comp_mse_list)[i+1]
        value = mse_list[index]
        min_dis_index_list.append(index)
        min_dis_value_list.append(value)
    return min_dis_index_list, min_dis_value_list, [x * -1 for x in comp_mse_list ]

def show_all_imgs(train_df, min_dis_index_list, shape, show = False):
    if not show:
        return None
    for i in range(len(min_dis_index_list)):
        min_index = min_dis_index_list[i]
        img = return_img(train_df,min_index, shape, show = False)
        if i == 0:
            img_whole = img
        else:
            img_whole = np.concatenate((img_whole, img), axis=1)
    if show:
        plt.imshow(img_whole)
        plt.colorbar()
        plt.show()

    return img_whole



train = train.sample(n=whole_num_sample,axis='rows', random_state = 0)
print(train.shape)
label = train["label"]
# print(train.isnull().any().sum())
# print(train.iloc[0])

# print(label.value_counts())
# print("train[1,:].shape = ", train.iloc[index1].values[1:].reshape(shape, shape, channel_num).shape)
# img1 =  train.iloc[index1].values[1:].reshape(shape, shape)
# img2 = train.iloc[index2].values[1:].reshape(shape, shape)

# plt.imshow(img1)
# plt.colorbar()
# plt.show()


############## LPIPS ##############
# lpips_value = lpips_index(train, index1, index2, shape)
# print('lpips_value:', lpips_value)
# min_distance_tsne = min_distance(train, index=1, metric = 'LPIPS')
# min_distance_tsne, min_distance_value_tsne = min_distance(train, index=1, metric = 'SSIM')
# show_all_imgs(train, min_distance_tsne, shape, show = show_image)
# print("min_distance: ", min_distance_tsne)

############## SSIM ##############
# ssim_value = ssim_index(train, index1, index2, shape)
# print('ssim', ssim_value)
import copy
train_df = copy.deepcopy(train) 
print(train.shape)
train.drop("label", axis=1, inplace=True)
print(train.shape)
train = StandardScaler().fit_transform(train)

if low_score == 'TSNE':
    tsne = TSNE(n_components = args.n_components, random_state=0)
    tsne_res = tsne.fit_transform(train)
elif low_score == 'UMAP':
    import umap.umap_ as umap
    reducer = umap.UMAP(n_components= args.n_components , random_state=0) # 42
    tsne_res = reducer.fit_transform(train)



############## TSNE_general ##############

all_csv_content = []
for i in range(run_file_num):
    print('##############',i ,  '##############')
    _ = return_img(train_df,i, shape, show = False, save = save_image_to_dir)
    min_distance_index_ssim, min_distance_value_ssim, high_sim_score = min_distance(train_df, index=i, metric = high_score)
    show_all_imgs(train_df, min_distance_index_ssim, shape, show = show_image)
    # print("min_distance: ", min_distance_tsne)
    min_distance_index_tsne, min_distance_value_tsne, low_sim_score = min_distance(train_df, index=i, metric = 'general', tsne_res = tsne_res)
    show_all_imgs(train_df, min_distance_index_tsne, shape, show = show_image)
    # print("min_distance: TSNE", min_distance_tsne)
    content = [i, min_distance_index_ssim, min_distance_value_ssim, min_distance_index_tsne, min_distance_value_tsne]
    sim_high[i,:] = high_sim_score
    sim_low[i,:]  = low_sim_score
    all_csv_content.append(content)

print(sim_high)
print(sim_low)
np.save(sim_high_filename, sim_high)
np.save(sim_low_filename, sim_low)

fields = ['index', 'high_nei', 'high_sim', 'low_nei', 'low_sim']
# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(all_csv_content)

data = pd.read_csv(filename)
for i in range(len(data)):
    for j in data.columns[1:]:
        data[j][i] = data[j][i][1:-1]
data.to_csv(filename)



# target_embed = tsne_res [search_index]



# plt.figure(figsize=(16,10))
# sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = label, palette = sns.hls_palette(10), legend = 'full')
# plt.plot(tsne_res[search_index,0], tsne_res[search_index,1],  color='r',markerfacecolor='red',marker='o',markersize=12) # , size = 2
# # plt.show()
# folder_name = 'figs/'
# plt.savefig(folder_name+'tsne_vanilia_specific.png',bbox_inches='tight',  pad_inches = 0)




# mse_list = (tsne_res [:, 0] - target_embed [0] ) ** 2  + (tsne_res [:, 1] - target_embed [1] ) ** 2 
# print("mse_list = ", mse_list)

# def min_distance_general():
#     index_list 
#     for i in range(num_neighbors):
#         min_index = np.argsort(mse_list)[i]
# for i in range(num_neighbors):
#     min_index = np.argsort(mse_list)[i]
#     img = return_img(train_df,min_index, shape, show = False)
#     if i == 0:
#         img_whole = img
#     else:
#         img_whole = np.concatenate((img_whole, img), axis=1)
# plt.imshow(img_whole)
# plt.colorbar()
# plt.show()
# plt.savefig('figs/mse_figures_specific.png',bbox_inches='tight',  pad_inches = 0)


 
# min_index = np.argsort(mse_list)[min_index_num]
# print("min_index:", min_index)

# img = return_img(train_df,min_index, shape, show = False)

# img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
# img1 = torch.zeros(1,3,64,64)
# if use_cuda:
#     img0 = img0.cuda()
#     img1 = img1.cuda()
# d = loss_fn_alex(img0, img1)









