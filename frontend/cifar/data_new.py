import pandas as pd
import numpy as np
import json

for k in ["UMAP", "TSNE"]:
    for j in ["cifar", "mnist"]:
        csv_path = "./cs765_project-master/" + j + "_SSIM_" + k + "_sample_100_num_neighbors_3.csv"
        nei_data = pd.read_csv(csv_path)
        for i in ["low", "SSIM"]:
            if i == "low":
                low = k
                file_path = "./cs765_project-master/" + j + "_SSIM_" + k + "_sample_100_num_neighbors_3_" + low + "_similarity.npy"
            else:
                file_path = "./cs765_project-master/" + j + "_SSIM_" + k + "_sample_100_num_neighbors_3_" + i + "_similarity.npy"
            sim_mat = np.load(file_path)
            data = []
            for idx in range(sim_mat.shape[0]):
                for idj in range(idx + 1, sim_mat.shape[0]):
                    item = {"src": idx, "dst": idj, "score": sim_mat[idx][idj]}
                    data.append(item)
            # data["scr"] = ",".join(src)
            # data["dst"] = ",".join(dst)
            # data["score"] = ",".join(score)
            with open("./json_data/" + j + "_SSIM_" + k + "_sample_100_num_neighbors_3_" + i + "_similarity.json", 'w') as f:
                json.dump(data, f, ensure_ascii=False)
            