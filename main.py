import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
	data = json.load(file)

try:
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)
except:
	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			tokens = nltk.word_tokenize(pattern)
			words.extend(tokens)
			docs_x.append(tokens)
			docs_y.append(intent["tag"])

			if intent["tag"] not in labels:
				labels.append(intent["tag"])

	words = [stemmer.stem(w.lower()) for w in words if w != "?"] #stems words
	words = sorted(list(set(words))) #removes duplicate elements

	labels = sorted(labels)


	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	#remembering the appearance (1 or 0) of a word in a sentence (creating a bag of words):
	for x, doc in enumerate(docs_x):
		bag = []

		tokens = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in tokens:
				bag.append(1)
			else:
				bag.append(0)


		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)

		with open("data.pickle", "wb") as f:
			pickle.dump((words, labels, training, output), f)

training = numpy.array(training)
output = numpy.array(output)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


#try:
	#model.load("model.tflearn")
#except:
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)


def chat():
	print("Start talking with the bot... (type quit to stop)")
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break

		results = model.predict([bag_of_words(inp, words)])[0]
		results_index = numpy.argmax(results)
		tag = labels[results_index]

		if results[results_index] > 0.7:
			for tg in data["intents"]:
				if tg["tag"] == tag:
					responses = tg["responses"]

			print(random.choice(responses))
		else:
			print("Sorry, I did not understand that. Try something else!")


chat()













