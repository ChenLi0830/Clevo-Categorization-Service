
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle 

from matplotlib import font_manager


# Step 6: Visualize the embeddings.
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches

  fontP = font_manager.FontProperties()
  fontP.set_family('SimHei')
  fontP.set_size(14)
  
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 fontproperties=fontP,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

if __name__=="__main__":
    
  print("loading word_idx_map data...")
  x = pickle.load(open("mr_folder/mr.p","rb"), encoding='latin1')
  revs, W, W2, word_idx_map, vocab, w2v = x[0], x[1], x[2], x[3], x[4], x[5]

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = W.shape[0]
  final_embeddings = W
  print(W.shape)
  word_idx_map.update({'未知': 0})


  reverse_dictionary = dict((v,k) for k,v in word_idx_map.items())
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  print(reverse_dictionary)
  print(len(reverse_dictionary))
  labels = [reverse_dictionary[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels, 'tsne.png')

