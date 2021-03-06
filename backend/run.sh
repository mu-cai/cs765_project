# generate data under all parameter settings
for low_score in UMAP TSNE
do
    {
    for dataset in mnist cifar
    do
        {
            for sample_per_class in 1 10 50 
            do
                {
                    python -W ignore backend.py --low_score $low_score --dataset $dataset --sample_per_class $sample_per_class
                }  &
            done
        }  &
    done
    }  &
done