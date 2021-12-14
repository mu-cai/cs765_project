
for low_score in 'UMAP' 'TSNE'
do
    {
        echo ${low_score}
        python -W ignore tsne_ssim.py --low_score ${low_score}
    } &
done