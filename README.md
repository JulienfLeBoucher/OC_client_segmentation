# O-list customer segmentation


In the [first notebook](https://nbviewer.org/github/JulienfLeBoucher/OC_clients_segmentation/blob/main/Olist_EDA_and_feature_engineering.ipynb#toc2_) of this project, I mainly merged data and created orders and clients summaries. I also implemented the well-know RFM segmentation technique.

Then I made use of unsupervised clustering algorithms (Kmeans, hierarchical, DBSCAN/OPTICS) and evaluated performance theoretically (silhouette, Calinski-Harabasz, Davies-Bouldin) on several input sets. I also compared and proposed interpretability of the best clusters obtained when they were relevant to a marketing team. This work is exposed in the 2 following notebooks : [here](https://nbviewer.org/github/JulienfLeBoucher/OC_clients_segmentation/blob/main/clustering_models.ipynb) and [there](https://nbviewer.org/github/JulienfLeBoucher/OC_clients_segmentation/blob/main/clustering_models_part2.ipynb).


Finally, in the [last notebook](https://nbviewer.org/github/JulienfLeBoucher/OC_clients_segmentation/blob/main/maintenance_simulations.ipynb), I simulated clients' information evolution through time in order to assess the need of a maintenance and determine an update frequency.

Watch my presentation if you want to get an overview.   