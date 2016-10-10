''' unused functions for now. may be useful in the future'''


# txt_to_cliques iterator. unused because we want to read over the 
# cliques twice once for positive and once for negative examples

def txt_to_cliques_iterator(shs_loc):
	'''
	# iterator version - unbecess here bc we use
	# creates a dictionary out of second hand songset 'cliques'
	# or groups of cover songs and returns the dict of cliques
	# based on their msd id
	'''
	shs = open(shs_loc)
	shs = itertools.islice(shs, 14, None)
	cliques = {}
	tempKey = None
	for ent in shs:
		ent = ent.replace('\n','')
		if ent[0] == '%':
			if tempKey:
				yield cliques
				cliques = {}
			tempKey = ent.lower()
			cliques[tempKey] = []
		else:
			cliques[tempKey].append(ent.split("<SEP>")[0]+'.mp3')
