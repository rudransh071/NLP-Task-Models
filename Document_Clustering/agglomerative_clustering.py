from scipy.cluster.hierarchy import ward, dendrogram
import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity

tfidf_matrix, titles, ranks, synopses, genres, vocab_frame, terms = preprocessing.get_data()
dist = 1 - cosine_similarity(tfidf_matrix)

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles)

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='False',      # ticks along the bottom edge are off
    top='False',         # ticks along the top edge are off
    labelbottom='False')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200)