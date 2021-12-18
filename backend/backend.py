from math import sqrt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import csv, os
np.random.seed(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--high_score', default='SSIM', type=str, help="score for high dimensional similarity, for example, 'SSIM' ")
parser.add_argument('--low_score', default='TSNE', type=str, help="core for high dimensional similarity, choose from [TSNE, UMAP]")
parser.add_argument('--dataset', default='mnist', type=str, help="data name: choose from [mnist, cifar]")
parser.add_argument('--sample_per_class', default=10, type=int, help="choose form 1-100")
parser.add_argument('--num_neighbors', default=3, type=int, help="choose from 1-10")
parser.add_argument('--n_components', default=2, type=int, help="choose from 1-10")
parser.add_argument('--image_save_path', default='./', type=str, help="the path to save images. I save it into public/html-s/cs765")
args = parser.parse_args()


dataset= args.dataset
high_score = args.high_score 
low_score = args.low_score

if high_score == 'LPIPS':
    import torch
    import torch.nn.functional as F
    import lpips
    use_cuda = torch.cuda.is_available()
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    if use_cuda:
        loss_fn_alex = loss_fn_alex.cuda()

index1 = 1
index2 = 2
scale = 255
sample_per_class = args.sample_per_class 
deep_input_size = 32
num_neighbors = args.num_neighbors # 3
show_image = False 
save_image_to_dir = not show_image
search_index = 2
min_index_num = 2
source_dir = 'source_data'
if dataset=='mnist':
    dataset_name = os.path.join( source_dir,  'source_mnist.csv' )
    channel_num = 1
elif dataset=='cifar':
    dataset_name= os.path.join( source_dir,   'source_cifar.csv' )
    channel_num = 3
train = pd.read_csv(dataset_name) 
train.head()
# print(train.shape[0])
label = train["label"]
num_class = len(label.value_counts())
whole_num_sample = num_class * sample_per_class 
run_file_num = whole_num_sample if not show_image else 1
shape = int(sqrt(train.iloc[index1].values[1:].shape[0] /channel_num )  )
print('Image_width: shape')
base_name = f"{dataset}_{high_score}_{low_score}_sample_{whole_num_sample}_num_neighbors_{num_neighbors}"
csv_folder = 'processed_data/csv'
filename = os.path.join( csv_folder,  base_name+ ".csv" )
sim_file_folder = 'processed_data/npy'
# if you want to save to html servers in cs machine, you can save it into following path
# home_dir =  os.environ['HOME']
# args.image_save_path =  os.path.join(home_dir, args.image_save_path)
# print(home_dir)

# path to save images
save_img_path = os.path.join(args.image_save_path, base_name)

if not os.path.exists(save_img_path) and not show_image:
    os.mkdir(save_img_path)
os.makedirs(sim_file_folder, exist_ok=True)


# initialize the similarity score array
save_similarity_score = True
if save_similarity_score:
    sim_high = np.zeros((whole_num_sample, whole_num_sample))
    sim_high_filename = os.path.join( sim_file_folder, base_name+ f"_{high_score}_similarity.npy" )
    sim_low = np.zeros((whole_num_sample, whole_num_sample))
    sim_low_filename =os.path.join( sim_file_folder,  base_name+ f"_{low_score}_similarity.npy" )


# reshape the images to make the visualization more general
def reshape_img(train,search_index):
    if channel_num==3:
        img1 =  train.iloc[search_index].values[1:].reshape(channel_num, shape, shape ) 
        img1 = np.transpose(img1, (1, 2, 0))
        img1 = img1.astype(np.uint8)
    elif channel_num ==1:
        img1 =  train.iloc[search_index].values[1:].reshape( shape, shape ) 
    return img1

# compute ssim
def ssim(img_org, img1, scale):
    return compare_ssim(img_org/scale, img1/scale,multichannel=True)
def ssim_index(train, index1, index2, shape):
    img1 = reshape_img(train,index1)
    img2 = reshape_img(train,index2)
    return ssim(img2, img1, scale)

# compute lpips
def lpips_index(train, index1, index2, shape):
    img1 =  train.iloc[index1].values[1:].reshape(shape, shape)/scale
    img2 =  train.iloc[index2].values[1:].reshape(shape, shape)/scale
    img1 =(torch.from_numpy(img1)).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)
    img2 =(torch.from_numpy(img2)).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)

    img1 = F.interpolate(img1, (deep_input_size, deep_input_size))
    img2 = F.interpolate(img2, (deep_input_size, deep_input_size))

    if use_cuda:
        img1 = img1.cuda()
        img2 = img2.cuda()
    return loss_fn_alex(img1.long(), img2.long()).item()

# show image
def return_img(train,search_index, shape, show = False, save = False):
    img1 =  reshape_img(train,search_index)
    if show:
        plt.imshow(img1)
        plt.colorbar()
        plt.show()
    if save:
        plt.imsave(os.path.join( save_img_path,  f'{search_index}.jpg'), img1)
    return img1

# get the min distance from a list
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


# visualize the images 
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


############## compute distance/similarity and find neighbors ##############

all_csv_content = []
for i in range(run_file_num):
    print('############## Sample ',i ,  '##############')
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

print('min: ', min(sim_high.reshape(run_file_num*run_file_num)), 'max,', max(sim_high.reshape(run_file_num*run_file_num)) )
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