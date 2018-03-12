'''
Class for the Deep HyperNEAT default genome and genes.

Felix Sosa
'''
import numpy as np
from itertools import count
from six_util import iteritems,itervalues,iterkeys
from random import choice
from activations import ActivationFunctionSet
from copy import deepcopy

# Mutation probabilities
node_add_prob = 0.3
node_delete_prob = 0.2
conn_add_prob = 0.5
conn_delete_prob = 0.5
weight_mutation_rate = 0.8
bias_mutation_rate = 0.7
inc_depth_prob = 0.2
inc_breadth_prob = 0

class Genome():

	def __init__(self, key):
		# Unique genome identifier
		self.key = key
		self.node_indexer = None
		# (key, gene) pairs for gene sets
		self.connections = {}
		self.nodes = {}
		# Genome information
		self.fitness = None
		self.num_inputs = 4
		self.num_outputs = 1
		self.num_layers = 2
		self.input_keys = [-i - 1 for i in range(self.num_inputs)]
		self.output_keys = range(self.num_outputs)
		self.cppn_tuples = [((1,0), (0,0))]#[((1,0),(2,0)), ((2,0),(0,0))] #
		self.activations = ActivationFunctionSet()
		self.configure()

	def configure(self):
		# Configure a new fully connected genome
		# print("\nGenome {} being configured".format(self.key))
		for input_id in self.input_keys:
			for output_id in self.output_keys:
				self.create_connection(input_id, output_id)
		for key, cppn_tuple in zip(self.output_keys,self.cppn_tuples):
			self.create_node('out',cppn_tuple,key)
	
	def copy(self, genome):
		# Copies the genes of another genome
		# print("Copying")
		self.node_indexer = deepcopy(genome.node_indexer)
		# print("Node Indexer Are Same: {}".format(genome.node_indexer is self.node_indexer))
		self.num_inputs = deepcopy(genome.num_inputs)
		# print("Num Inputs Are Same: {}".format(genome.num_inputs is self.num_inputs))
		self.num_outputs = deepcopy(genome.num_outputs)
		# print("Num Outputs Are Same: {}".format(genome.num_outputs is self.num_outputs))
		self.input_keys = deepcopy(genome.input_keys)
		# print("Input Keys Are Same: {}".format(genome.input_keys is self.input_keys))
		# print("Copied Genome {} Output Keys: {}".format(genome.key, genome.output_keys))
		self.output_keys = [x for x in genome.output_keys]
		# print("Output Keys Are Same: {}".format(genome.output_keys is self.output_keys))
		self.cppn_tuples = deepcopy(genome.cppn_tuples)
		self.num_layers = deepcopy(genome.num_layers)
		# Nodes
		# print("Copied Genome {} Nodes: {}".format(genome.key, genome.nodes))
		for node_copy in genome.nodes.values():
			node_to_add = NodeGene(node_copy.key,node_copy.type,
								   node_copy.activation, node_copy.cppn_tuple)
			node_to_add.bias = node_copy.bias
			self.nodes[node_to_add.key] = node_to_add
		# Connections
		for conn_copy in genome.connections.values():
			conn_to_add = ConnectionGene(conn_copy.key, conn_copy.weight)
			self.connections[conn_to_add.key] = conn_to_add
		# self.connections = genome.connections
		# print("Copied Nodes: {}".format(self.nodes))
		# print("Done Copying")

	def create_connection(self, source_key, target_key, weight=None):
		# Create a new connection gene
		if not weight:
			weight = np.random.uniform(-1,1)
			# print(weight)
		new_conn = ConnectionGene((source_key,target_key), weight)
		# print(new_conn.weight)
		self.connections[new_conn.key] = new_conn
		return new_conn

	def create_node(self,node_type='hidden',cppn_tuple=None,key=None):
		# Create a new node
		activation_key = np.random.choice(self.activations.functions.keys())
		activation = self.activations.get(activation_key)
		new_node_key = self.get_new_node_key() if key == None else key
		new_node = NodeGene(new_node_key, node_type, activation, cppn_tuple)
		self.nodes[new_node.key] = new_node
		# if node_type == 'out':
			# print('\n{} New Output Node {}'.format(self.key, new_node.key))
			# print('{} Nodes in Genome: {}'.format(self.key, self.nodes))
			# print('{} Current Output Nodes: {}'.format(self.key, self.output_keys))
		return new_node

	def mutate(self):
		# Mutate genome
		if np.random.uniform() < node_add_prob:
			self.mutate_add_node()
		if np.random.uniform() < node_delete_prob:
			self.mutate_delete_node()
		if np.random.uniform() < conn_add_prob:
			self.mutate_add_connection()
		if np.random.uniform() < conn_delete_prob:
			self.mutate_delete_connection()
		if np.random.uniform() < inc_depth_prob:
			self.mutate_increment_depth()
		if np.random.uniform() < inc_breadth_prob:
			self.mutate_increment_breadth()

		# Mutate connection genes.
		for conn_gene in self.connections.values():
			conn_gene.mutate()
		# Mutate node genes (bias, response, etc.).
		for node_gene in self.nodes.values():
			node_gene.mutate()

	def mutate_add_node(self):
		# Add new node to the genome
		# Choose connection to split
		if self.connections:
			idx = np.random.choice(range(len(self.connections)))
			conn_to_split = list(self.connections.keys())[idx]
		else:
			return
		# Create new hidden node and add to genome
		new_node = self.create_node()
		self.nodes[new_node.key] = new_node
		# Get weight from old connection
		old_weight = self.connections[conn_to_split].weight
		# Delete connection from genome
		del self.connections[conn_to_split]
		# Create i/o connections for new node
		i, o = conn_to_split
		self.create_connection(i, new_node.key, 1.0)
		self.create_connection(new_node.key, o, old_weight)

	def mutate_add_connection(self):
		# Add a new connection to the genome
		# Gather possible target nodes and source nodes
		if not self.nodes:
			return
		possible_targets = list(iterkeys(self.nodes))
		target_key = choice(possible_targets)
		possible_sources = possible_targets + self.input_keys
		source_key = choice(possible_sources)

		# Ensure connection isn't duplicate
		if (source_key,target_key) in self.connections:
			self.connections[(source_key,target_key)].enabled = True
			return

		# Don't allow connections between two output nodes
		if source_key in self.output_keys and target_key in self.output_keys:
			return

		new_conn = self.create_connection(source_key, target_key)
		self.connections[new_conn.key] = new_conn

	def mutate_delete_node(self):
		# Delete a node
		available_nodes = [k for k in iterkeys(self.nodes) if k not in self.output_keys]
		if not available_nodes:
			return

		# Choose random node to delete
		del_key = np.random.choice(available_nodes)
		# Iterate through all connections and find connections to node
		conn_to_delete = set()
		for k, v in iteritems(self.connections):
			if del_key in v.key:
				conn_to_delete.add(v.key)

		for i in conn_to_delete:
			del self.connections[i]

		# Delete node key
		del self.nodes[del_key]
		return del_key

	def mutate_delete_connection(self):
		# Delete a connection
		if self.connections:
			idx = np.random.choice(range(len(self.connections)))
			key = list(self.connections.keys())[idx]
			del self.connections[key]
	
	def mutate_increment_depth(self):
		# Add CPPNON to increment depth of Substrate
		# Create CPPN tuple
		source_layer = self.num_layers
		target_layer, target_sheet, source_sheet = 0, 0, 0
		cppn_tuple = ((source_layer, source_sheet),
					  (target_layer,target_sheet))
		# Adjust tuples for previous CPPNONs
		for key in self.output_keys:
			tup = self.nodes[key].cppn_tuple
			if tup[1] == (0,0):
				self.nodes[key].cppn_tuple = (tup[0], 
											  (source_layer,
											   source_sheet))
		# Create two new gaussian nodes
		gauss_1_node = self.create_node()
		gauss_1_node.activation = self.activations.get('dhngauss')
		gauss_1_node.bias = 0.0
		gauss_2_node = self.create_node()
		gauss_2_node.activation = self.activations.get('dhngauss')
		gauss_2_node.bias = 0.0
		gauss_3_node = self.create_node()
		gauss_3_node.activation = self.activations.get('dhngauss2')
		gauss_3_node.bias = 0.0
		# Create new CPPN Output Node (CPPNON)
		output_node = self.create_node('out', cppn_tuple)
		output_node.activation = self.activations.get('linear')
		output_node.bias = 0.0
		# Add new CPPNON key to list of output keys in genome
		self.num_outputs += 1
		self.num_layers += 1
		self.output_keys.append(output_node.key)
		# Add connections
		# x1 to gauss 1
		self.create_connection(self.input_keys[0], 
							gauss_1_node.key, -1.0)
		# x2 to gauss 1
		self.create_connection(self.input_keys[2], 
							gauss_1_node.key, 1.0)
		# y1 to gauss 2
		self.create_connection(self.input_keys[1], 
							gauss_2_node.key, -1.0)
		# y2 to gauss 2
		self.create_connection (self.input_keys[3], 
							gauss_2_node.key, 1.0) 
		# Gauss 1 to gauss 3
		self.create_connection(gauss_1_node.key, 
							gauss_3_node.key, 1.0)
		# Gauss 2 to gauss 3
		self.create_connection(gauss_2_node.key, 
							gauss_3_node.key, 1.0)
		# Gauss 3 to CPPNON
		self.create_connection(gauss_3_node.key,
							output_node.key,1.0)

	def mutate_increment_breadth(self):
		# Add CPPNON to increment breadth of Substrate
		pass
	
	def get_new_node_key(self):
		# Returns new node key
		if self.node_indexer is None:
			self.node_indexer = count(max(self.output_keys)+1)
		new_id = next(self.node_indexer)
		assert new_id not in self.nodes
		return new_id

class NodeGene():

	def __init__(self,key,node_type,activation,cppn_tuple):
		self.type = node_type
		self.key = key
		self.bias = np.random.uniform(-1,1)
		self.activation = activation
		self.response = 1.0
		self.cppn_tuple = cppn_tuple

	def mutate(self):
		# Mutate attributes of node gene
		if np.random.uniform() < bias_mutation_rate:
			self.bias += np.random.uniform(-0.5,0.5)
		# self.response += np.random.uniform(-0.1,0.1)

class ConnectionGene():

	def __init__(self,key,weight):
		self.key = key
		self.weight = weight
		self.enabled = True

	def mutate(self):
		# Mutate attributes of connection gene
		if np.random.uniform() < weight_mutation_rate:
			self.weight += np.random.uniform(-5,5)