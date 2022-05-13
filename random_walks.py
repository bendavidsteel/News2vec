
import collections
import random

from numba import njit
import numpy as np
import networkx as nx
import tqdm

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = np.sort(np.fromiter(G.neighbors(cur), dtype=int))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print ('Walk iteration:')
		for walk_iter in range(num_walks):
			print (str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		dst_neighbours = np.sort(np.fromiter(G.neighbors(dst), dtype=int))
		for dst_nbr in dst_neighbours:
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		unnormalized_probs = np.array(unnormalized_probs)
		normalized_probs =  unnormalized_probs / np.sum(unnormalized_probs)

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in tqdm.tqdm(G.nodes(), total=G.number_of_nodes()):
			neighbours = np.sort(np.fromiter(G.neighbors(node), dtype=int))
			unnormalized_probs = np.array([G[node][nbr]['weight'] for nbr in neighbours]).astype(float)
			normalized_probs = unnormalized_probs / np.sum(unnormalized_probs)
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		num_edges = G.number_of_edges()
		if is_directed:
			for edge in tqdm.tqdm(G.edges(), total=num_edges):
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in tqdm.tqdm(G.edges(), total=num_edges):
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = probs.shape[0]
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	p = probs * K
	smaller = collections.deque(np.nonzero(p < 1.0)[0])
	larger = collections.deque(np.nonzero(p >= 1.0)[0])

	while smaller and larger:
		small = smaller.popleft()
		large = larger.popleft()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]
