
import random
import sys
import os

OUT_DIR = 'output'

def read_input(in_file_name):
	with open(in_file_name, 'r') as myfile:
		data=myfile.read()
		data=data.lower()
		data=data.replace('(', '')
		data=data.replace(')', '')
		data=data.replace('.', '')
		data=data.replace(',', '')
		data=data.replace('!', '')
		data=data.replace(';', '')
		data=data.replace("'", '')
		data=data.replace('"', '')
		data=data.replace(' - ', ' ')
		data=data.replace('\n', ' \n ')
	return data

def markov_chain(input_text, prev_words):
	words = input_text.split(' ')
	markov_map = dict()
	for i in range(len(words) - prev_words):
		key_tup = tuple(words[i : i + prev_words])
		next_word = words[i + prev_words]
		if key_tup not in markov_map:
			markov_map[key_tup] = [next_word]
		else:
			markov_map[key_tup].append(next_word)
	return markov_map

def generate():
	temp, input_file, prev_words, song_length = sys.argv
	lyric_string = read_input(input_file)
	model = markov_chain(lyric_string, int(prev_words))
	first_words = random.choice(list(model.keys()))
	window = list(first_words)
	song_lyrics_list = list(window)
	while len(song_lyrics_list) < int(song_length) or next_word != '\n':
		#print(window)
		next_word = random.choice(model[tuple(window)])
		song_lyrics_list.append(next_word)
		window.append(next_word)
		window = window[1:]
		#print(window)
		#print(next_word)
	song_lyrics = ' '.join(song_lyrics_list).replace('\n ', '\n')
	out_file_name = 'markov_' + input_file.replace('.txt', '') + '_' + prev_words + '_' + song_length + '.txt'
	with open(os.path.join(OUT_DIR, out_file_name), 'a') as outfile:
		outfile.write(song_lyrics)
		#outfile.write('\n')
		outfile.write('############################\n')


generate()

