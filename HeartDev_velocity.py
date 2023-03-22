
import numba  
import anndata
import scvelo as scv
import pandas as pd
import numpy as np
import matplotlib as plt



# Genes of interest
gois1 =["Hand1",'Nkx2-5', 'Isl1', 'Mesp1', 'Pmp22', 'Tnnt2', 'Tnni1','Tnni3','Pdgfra','Fgf8']
gois2 =['Bmp4','Hoxb1','Hoxb6','Krt8','Myl2','Myl7','Myh6','Myh7','Mef2c','Slit2']
gois3 =['Slit1','Robo2','Tbx18','EGFP','Kdr','Fst','Tcf15','Hcn4','Foxf1','Aldh1a2']
gois4 =['Sox9','Dlx5','Kcna5','Irx4','Apobec2','Ckb','Myl4','Hopx','Spink1','Ttr'] 
gois5 =['Apom','Apoe','Meis2','Cdk1nc',"Gata4","Gata6","Tbx5","Tbx20","Sfpr1","Sfpr5"] 
gois6 =["Lbh", "Stard10", "Vsnl1", "Mest","Pln", "Rspo3", "Nfatc1", "Wt1", "Shox2", "Sfr"]
gois7 =["Axin2", "Tbx3", "Nppa", "Bmp2", "Foxh1", "Fgf10", "Gata5", "Ttn","Acta2", "Actn2"]
gois8 =["Nr2f2", "Sparc", "Fbxl22", "Apoa1", "Phlda2", "Al4c", "Mid1ip1", "Mtus2", "Plac8", "Cnn1", "Ankrd1", "Csrp3"]



### Data prep

## Load csv file containing cell IDs corresponding to Seurat analysis-filtered cells
sample_obs = pd.read_csv("cellID_obs_loomIDs.csv")
# Load data in loom format 
adata = scv.read("../data/combined_data.loom")
# Keep only predetemined cells
adata = adata[np.isin(adata.obs.index, sample_obs["x"])]

## Load and add Seurat analysis-created UMAP and cluster info
X_umap = scv.load('umap_coords.csv', index_col=0)
adata.obs['X_umap'] = X_umap.loc[adata.obs_names].values
cell_clusters = scv.load("clusters.csv", index_col=0)
adata.obs['cell_clusters'] = cell_clusters.loc[adata.obs_names].values

## Preprocess data
scv.pp.filter_and_normalize(adata)
scv.pp.moments(adata)




### Basic velocity analysis


## Calculate velocity
scv.tl.velocity(adata, mode = "stochastic")
scv.tl.velocity_graph(adata)


## Plots

# Create velocity stream plots
scv.pl.velocity_embedding_stream(adata, basis='umap', save='stream_umap_clusters.png', color='cell_clusters', palette = pal, dpi=300, figsize = (9,6))

# Create phase portraits, velocity and expression plots for genes of interest
scv.pl.velocity(adata, var_names=gois1, save="gene.plots_1_clustercols.png", size=30, color = 'cell_clusters', palette = pal, dpi=300, figsize = (9,11))
scv.pl.velocity(adata, var_names=gois2, save="gene.plots_2_clustercols.png", size=30, color = 'cell_clusters', palette = pal, dpi=300, figsize = (9,11))
scv.pl.velocity(adata, var_names=gois3, save="gene.plots_3_clustercols.png", size=30, color = 'cell_clusters', palette = pal, dpi=300, figsize = (9,11))
scv.pl.velocity(adata, var_names=gois4, save="gene.plots_4_clustercols.png", size=30, color = 'cell_clusters', palette = pal, dpi=300, figsize = (9,11))
scv.pl.velocity(adata, var_names=gois5, save="gene.plots_5_clustercols.png", size=30, color = 'cell_clusters', palette = pal, dpi=300, figsize = (9,11))
scv.pl.velocity(adata, var_names=gois6, save="gene.plots_6_clustercols.png", size=30, color = 'cell_clusters', palette = pal, dpi=300, figsize = (9,11))
scv.pl.velocity(adata, var_names=gois7, save="gene.plots_7_clustercols.png", size=30, color = 'cell_clusters', palette = pal, dpi=300, figsize = (9,11))
scv.pl.velocity(adata, var_names=gois8, save="gene.plots_8_clustercols.png", size=30, color = 'cell_clusters', palette = pal, dpi=300, figsize = (9,11))


## Velocity driving genes

# Differential velocity expression for each cluster
#(run differential velocity t-test)
scv.tl.rank_velocity_genes(adata, groupby='cell_clusters', min_corr=.3)
df = scv.DataFrame(adata.uns['rank_velocity_genes']['names'])
# Save to csv
df.to_csv("./velocity_gene_rank_by_cluster.csv")


## Differentiation speed and confidence

# Calculate
scv.tl.velocity_confidence(adata)

# Plot
keys = 'velocity_length', 'velocity_confidence'
scv.pl.scatter(adata, c=keys, cmap='coolwarm', perc=[5, 95], save='differentiation_speed_conf_ra.pdf', dpi=300, figsize = (8,6))
# Save as csv
df = adata.obs.groupby('cell_clusters')[keys].mean().T
df.style.background_gradient(cmap='coolwarm', axis=1)
df.to_csv("./velocity_differentiation_speed_conf_by_clusters.csv")


## Pseudotime

# Plot velocity-inferred cell-to-cell transitions
scv.pl.velocity_graph(adata, threshold=0.5, color='cell_clusters', save='cell_transitions_clustercols_ra.pdf', dpi=300, figsize = (8,6))

# Calculate velocity pseudotime
scv.tl.velocity_pseudotime(adata)
# Plot
scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot', save='pseudotime_ra.pdf', dpi=300, figsize = (8,6))





### Dynamic modeling


## Transcriptional state and cell-internal latent time 

# Run likelihood-based expectation-maximization framework and estimate parameters of reaction rates and latent cell-specific variables
scv.tl.recover_dynamics(adata)

# Recalculate velocity based on dynamic modeling
scv.tl.velocity(adata, mode='dynamical')
scv.tl.velocity_graph(adata)

# Create dynamic velocity stream plots
scv.pl.velocity_embedding_stream(adata, basis='umap', save='stream_umap_clusters_dynamic.pdf', color='cell_clusters', palette = pal, dpi=300, figsize = (9,6))


## Latent time

# Approximate real time experienced by cells during differentiation
scv.tl.latent_time(adata)
# Plot
scv.pl.scatter(adata, color='latent_time', color_map='gnuplot', size=30, save="stream_umap_latenttime.pdf")


## Dynamic driver genes

# Calculate likelihoods in the dynamic model
top_genes = adata.var['fit_likelihood'].sort_values(ascending=False).index[:300]
# Save top likelihood genes
df = scv.get_df(top_genes)
df.to_csv("./top300-likelihood-genes_latenttime.csv")


# Plot heatmap for driver genes
scv.pl.heatmap(adata, var_names=top_genes, sortby='latent_time', col_color='cell_clusters', n_convolve=100, save="heatmap_latenttime.png", figsize = (9,7))
# Plot heatmap for genes of interest
scv.pl.heatmap(adata, var_names=gois, sortby='latent_time', col_color='cell_clusters', n_convolve=100, yticklabels=True, save="heatmap_gois_latenttime.png", figsize = (9,12))



# Plot dynamic phase portraits for driver genes and genes of interest
scv.pl.scatter(adata, basis=top_genes[:14], ncols=7, frameon=False, size=50, color="cell_clusters", save="dynamic_phaseplot_driver_genes.pdf")

scv.pl.scatter(adata, gois1, ncols=5, frameon=False, color="cell_clusters", save="dynamic_phaseplot_gois_part1.png", size=50)
scv.pl.scatter(adata, gois2, ncols=5, frameon=False, color="cell_clusters", save="dynamic_phaseplot_gois_part2.png", size=50)
scv.pl.scatter(adata, gois3, ncols=5, frameon=False, color="cell_clusters", save="dynamic_phaseplot_gois_part3.png", size=50)
scv.pl.scatter(adata, gois4, ncols=5, frameon=False, color="cell_clusters", save="dynamic_phaseplot_gois_part4.png", size=50)
scv.pl.scatter(adata, gois5, ncols=5, frameon=False, color="cell_clusters", save="dynamic_phaseplot_gois_part5.png", size=50)
scv.pl.scatter(adata, gois6, ncols=5, frameon=False, color="cell_clusters", save="dynamic_phaseplot_gois_part6.png", size=50)
scv.pl.scatter(adata, gois7, ncols=5, frameon=False, color="cell_clusters", save="dynamic_phaseplot_gois_part7.png", size=50)
scv.pl.scatter(adata, gois8, ncols=5, frameon=False, color="cell_clusters", save="dynamic_phaseplot_gois_part8.png", size=50)


# Plot expression over latent time for driver genes and genes of interest
scv.pl.scatter(adata, x='latent_time', y=top_genes[:14], ncols=7, frameon=False, size=50, color="cell_clusters", save="expression_latenttime_driver_genes.pdf")

scv.pl.scatter(adata, x='latent_time', y=gois1, ncols=5, frameon=False, color="cell_clusters", save="expression_latenttime_gois_part1.png", size=50)
scv.pl.scatter(adata, x='latent_time', y=gois2, ncols=5, frameon=False, color="cell_clusters", save="expression_latenttime_gois_part2.png", size=50)
scv.pl.scatter(adata, x='latent_time', y=gois3, ncols=5, frameon=False, color="cell_clusters", save="expression_latenttime_gois_part3.png", size=50)
scv.pl.scatter(adata, x='latent_time', y=gois4, ncols=5, frameon=False, color="cell_clusters", save="expression_latenttime_gois_part4.png", size=50)
scv.pl.scatter(adata, x='latent_time', y=gois5, ncols=5, frameon=False, color="cell_clusters", save="expression_latenttime_gois_part5.png", size=50)
scv.pl.scatter(adata, x='latent_time', y=gois6, ncols=5, frameon=False, color="cell_clusters", save="expression_latenttime_gois_part6.png", size=50)
scv.pl.scatter(adata, x='latent_time', y=gois7, ncols=5, frameon=False, color="cell_clusters", save="expression_latenttime_gois_part7.png", size=50)
scv.pl.scatter(adata, x='latent_time', y=gois8, ncols=5, frameon=False, color="cell_clusters", save="expression_latenttime_gois_part8.png", size=50)


