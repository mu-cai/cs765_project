sample_per_class_list=(10 50 1)
for low_score in UMAP TSNE
do
    {
        for dataset in mnist cifar
        do
            {
                for sample_per_class in sample_per_class_list
                do
                    {
                        python -W ignore tsne_ssim.py --low_score $low_score --dataset $dataset --sample_per_class $sample_per_class
                    } 
            } 
    } 
done