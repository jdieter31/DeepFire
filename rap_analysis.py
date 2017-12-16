import PyDictionary
dictionary=PyDictionary.PyDictionary()

def get_multiparts(parts_of_speech):
	two_parts = []
	three_parts = []
	for i in range(len(parts_of_speech) - 2):
		if parts_of_speech[i] != None and parts_of_speech[i + 1] != None:
			two_parts.append((parts_of_speech[i], parts_of_speech[i + 1]))
			if parts_of_speech[i + 2] != None:
				three_parts.append((parts_of_speech[i], parts_of_speech[i + 1], parts_of_speech[i + 2]))
	return two_parts, three_parts

def test_common_constructions(lines):
	words = []
	for line in lines:
		words += line.split(" ")
	
	parts_of_speech = []

	for word in words:
		print("Handling " + word)
		try:
			meaning = dictionary.meaning(word)
		except IndexError:
			pass
		if meaning == None:
			parts_of_speech.append(None)
		else:
			parts_of_speech.append(list(dictionary.meaning(word).keys()))


	two_parts, three_parts = get_multiparts(parts_of_speech)

	print(parts_of_speech)
	print(test_adjective_noun_constructions(two_parts))
	print(test_noun_verb_constructions(two_parts))
	print(test_adverb_verb_constructions(two_parts))
	print(test_noun_verb_noun_constructions(three_parts))

# Tests the proportion of times a noun follows an adjective
# Skips adjectives
def test_adjective_noun_constructions(two_parts):
	count = 0
	valid = 0
	for two_part in two_parts:
		if "Adjective" in two_part[0] :
			count += 1
			if "Adjective" in two_part[1]:
				count -= 1
			elif "Noun" in two_part[1]:
				valid += 1
	if count == 0:
		return None
	else:
		return float(valid) / float(count)



# Tests the proportion of times a verb follows a noun
# Skips adverbs
def test_noun_verb_constructions(two_parts):
	count = 0
	valid = 0
	for two_part in two_parts:
		if "Noun" in two_part[0] :
			count += 1
			if "Adverb" in two_part[1]:
				count -= 1
			elif "Verb" in two_part[1]:
				valid += 1
	if count == 0:
		return None
	else:
		return float(valid) / float(count)

# Counts the number of times verbs appear by adverbs
def test_adverb_verb_constructions(two_parts):
	count = 0
	valid = 0
	for two_part in two_parts:
		if "Adverb" in two_part[0]:
			count += 1
			if "Verb" in two_part[1]:
				valid += 1
		if "Adverb" in two_part[1]:
			count += 1
			if "Verb" in two_part[0]:
				valid += 1
	if count == 0:
		return None
	else:
		return float(valid) / float(count)

# Tests the number of times a noun follows a noun verb construction
# Skips adjectives
def test_noun_verb_noun_constructions(three_parts):
	count = 0
	valid = 0
	for three_part in three_parts:
		if "Noun" in three_part[0] and "Verb" in three_part[1]:
			count += 1
			if "Noun" in three_part[2]:
				valid += 1
			elif "Adjective" in three_part[2]:
				count -= 1
	if count == 0:
		return None
	else:
		return float(valid) / float(count)

# f = open("data/rap.txt")
# f = open("output/markov_data/rap_2_500.txt")
# f = open("output/markov_data/rap_3_500.txt")
# f = open("lstm2layer_output.txt")
f = open("lstm1layer_output.txt")


t = f.readlines()
test_common_constructions(t[0:50])
