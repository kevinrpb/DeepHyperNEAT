'''
Set of functions for reporting status of an evolutionary run.

NOTE: Only meant for XOR at the moment. Working on generalizing to any task.
'''
from deep_hyperneat.util import iteritems,itervalues,iterkeys
from deep_hyperneat.phenomes import FeedForwardCPPN as CPPN
from deep_hyperneat.phenomes import FeedForwardSubstrate as Substrate
from deep_hyperneat.decode import decode
import seaborn
import matplotlib.pyplot as plt

xor_substrate = {
	'in_dims': [1,2],
	'sh_dims': [1,3],
	'o_dims': 1
}
xor_inputs = [(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]
xor_outputs = [0.0, 1.0, 1.0, 0.0]

def report_output(pop, X=None, Y=None, substrate=None):
	'''
	Reports the output of the current champion for the xor task.

	pop -- population to be reported
	'''
	if X is None:
		X = xor_inputs
	if Y is None:
		Y = xor_outputs
	if substrate is None:
		substrate = xor_substrate

	genome = pop.best_genome
	cppn = CPPN.create(genome)
	substrate = decode(cppn,substrate['in_dims'],substrate['o_dims'],substrate['sh_dims'])
	sum_square_error = 0.0
	print("\n=================================================")
	print("\tChampion Output at Generation: {}".format(pop.current_gen))
	print("=================================================")
	for inputs, expected in zip(X, Y):
		print("Input: {}\nExpected Output: {}".format(inputs,expected))
		inputs = inputs + (1.0,)
		actual_output = substrate.activate(inputs)[0]
		sum_square_error += ((actual_output - expected)**2.0)/4.0
		print("Actual Output: {}\nLoss: {}\n".format(actual_output,sum_square_error))
	print("Total Loss: {}".format(sum_square_error))

def report_fitness(pop):
	'''
	Report average, min, and max fitness of a population

	pop -- population to be reported
	'''
	avg_fitness = 0
	# Find best genome in current generation and update avg fitness
	for genome in itervalues(pop.population):
		avg_fitness += genome.fitness
	print("\n=================================================")
	print("\t\tGeneration: {}".format(pop.current_gen))
	print("=================================================")
	print("Best Fitness \t Avg Fitness \t Champion")
	print("============ \t =========== \t ========")
	print("{:.2f} \t\t {:.2f} \t\t {}".format(pop.best_genome.fitness,
		  avg_fitness/pop.size,pop.best_genome.key))
	print("=================================================")
	print("Max Complexity \t Avg Complexity")
	print("============ \t =========== \t ========")
	print("{} \t\t {}".format(None, pop.avg_complexity))

def report_species(species_set, generation):
	'''
	Reports species statistics

	species_set -- set contained the species
	generation  -- current generation
	'''
	print("\nSpecies Key \t Fitness Mean/Max \t Sp. Size")
	print("=========== \t ================ \t ========")
	for species in species_set.species:
		# print("{} \t\t {:.2} / {:.2} \t\t {}".format(species,
		# 	species_set.species[species].fitness,
		# 	species_set.species[species].max_fitness,
		# 	len(species_set.species[species].members)))
		print(species,
			species_set.species[species].fitness,
			species_set.species[species].max_fitness,
			len(species_set.species[species].members))

def plot_fitness(x,y,filename, plot_settings={}):
	# Check used keys and set to none if not present
	if 'ylim' not in plot_settings:
		plot_settings['ylim'] = None

	plt.plot(x,y)
	plt.ylabel("Fitness")
	plt.xlabel("Generation")
	plt.tight_layout()
	plt.ylim(plot_settings['ylim'])
	plt.savefig(filename)
	plt.clf()
