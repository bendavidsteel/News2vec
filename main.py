import argparse
import pickle
import os

import numpy as np
import networkx as nx

import random_walks
from news2vec import newsfeature2vec

def parse_args():
	'''
	Parses the News2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run News2vec.")

	parser.add_argument('--input', nargs='?', default='data/0721',
						help='Directory with input files')

	parser.add_argument('--output', nargs='?', default='emb/0721.emb',
						help='Embeddings path')

	parser.add_argument('--include', nargs='?', default=True,
						help='Boolean including element nodes')

	parser.add_argument('--dimensions', type=int, default=128,
						help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=100,
						help='Length of walk per source. Default is 100.')

	parser.add_argument('--num-walks', type=int, default=5,
						help='Number of walks per source. Default is 5.')

	parser.add_argument('--window-size', type=int, default=5,
						help='Context size for optimization. Default is 5.')

	parser.add_argument('--num-iterations', type=int, default=None,
						help='Number of iterations to train for. Default is proportional to dataset size.')

	parser.add_argument('--p', type=float, default=1,
						help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
						help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is weighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=True)

	parser.add_argument('--directed', dest='directed', action='store_true',
						help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.add_argument('--checkpoint-path',
						help='Model checkpoints path.')

	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph(edge_filepath):
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(edge_filepath, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(edge_filepath, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def map_news(walks, map_filepath):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''

	map_dict = {}
	with open(map_filepath, 'r',encoding='utf-8') as f:
		for l in f:
			l = l.strip('\n').split(' ')
			map_dict[l[0]] = l[1]

	article_walks = []
	for walk in walks:
		article_walks += list(map(lambda x: map_dict[str(x)], walk))
	
	return article_walks

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''

	dataset_dir = args.input
	dataset_files = os.listdir(dataset_dir)

	if os.path.exists(os.path.join(dataset_dir, 'dataset.pickle')):
		with open(os.path.join(dataset_dir, 'dataset.pickle'), 'rb') as handle:
			all_article_walks = pickle.load(handle)

	else:
		all_article_walks = []
		for edge_file in [dataset_file for dataset_file in dataset_files if 'edgelist' in dataset_file]:
			print(f"Loading graph from file {edge_file}")

			edge_filepath = os.path.join(dataset_dir, edge_file)
			nx_G = read_graph(edge_filepath)
			G = random_walks.Graph(nx_G, args.directed, args.p, args.q)
			print("Preprocess transition probabilities")
			G.preprocess_transition_probs()
			print("Simulate walks")
			walks = G.simulate_walks(args.num_walks, args.walk_length)
			
			print("Map news")
			map_filepath = edge_filepath.replace("s.edgelist", "_nodes.map")
			article_walks = map_news(walks, map_filepath)

			all_article_walks += article_walks

		with open(os.path.join(dataset_dir, 'dataset.pickle'), 'wb') as handle:
			pickle.dump(all_article_walks, handle, protocol=pickle.HIGHEST_PROTOCOL)


	num_iters = int(len(all_article_walks)) if args.num_iterations is None else args.num_iterations
	print("Learn embeddings")
	newsfeature2vec(all_article_walks,args.output,include=args.include,skip_window=args.window_size,iter=num_iters, save_path=args.checkpoint_path)

if __name__ == "__main__":
	args = parse_args()
	main(args)
