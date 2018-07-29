#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Latent Dirichlet Allocation:  Implementation of the algorithm in Python
Author: Samyadeep Basu               
Research Area: Probabilistic Graphical Models / Topic Modelling    
Training Method: Collapsed Gibbs Sampling                                    """



""" Algorithm Description: 
          """

          
#Libraries
import numpy as np 
import operator

################################################# Algorithmic Steps ###################################################

#Function to initialise the matrices before learning begins 
def initialise_matrices(docs, doc_word_topic, theta, phi, word_count_topic):
	for counter, document in enumerate(docs):
		#For updating doc_word_topic
		temp = []
		for word in document:
			#Number of times 'word' has been observed with a given topic : Will come from phi 
			t_1 = phi[:,word]

			#Number of times a given topic is observed within a word of given 'document' : Will come from theta
			t_2 = theta[counter]

			probabilities = t_1 * t_2 / word_count_topic 

			#Normalised probabilities
			probabilities = probabilities / probabilities.sum()

			#Sample from multinomial distribution - Only single trial / categorical
			topic = np.random.multinomial(1,probabilities).argmax()

			#Update existing tables
			theta[counter, topic] += 1

			phi[topic, word] += 1

			word_count_topic[topic] += 1

			temp.append(topic)


		doc_word_topic.append(temp)

	return 


#Function to compute the perplexity : Lower the perplexity, better is the model
def compute_perplexity(alpha, docs, doc_word_topic, theta, phi, word_count_topic):
	perplex = 0 
	phi_normalised = phi / word_count_topic[:,np.newaxis]

	N = 0
	for counter, doc in enumerate(docs):
		for word in doc: 
			theta_normalised = theta[counter] / (len(docs[counter]) )
			phi_new = phi_normalised[:,counter] 
			perplex -= np.log(np.inner(theta_normalised,phi_new))
			

		N += len(doc)
	
	final_perplexity = np.exp(perplex/N)

	return final_perplexity


#Function to infer and learn and update parameters
def infer_learn(docs, doc_word_topic, theta, phi, word_count_topic):
	for counter, doc in enumerate(docs):
		for word_counter, word in enumerate(doc):
	
			#Get the topic of the current word
			topic_current_word = doc_word_topic[counter][word_counter]

			phi[topic_current_word, word] -=1

			#Remove the topic from the theta 
			theta[counter,topic_current_word] -= 1

			#Remove count from main topic_count list
			word_count_topic[topic_current_word] -=1 

			#Sample a new topic from distribution and create an update for the word 
			t_1 = phi[:, word]
			t_2 = theta[counter]

			probabilities = t_1 * t_2 / word_count_topic

			#Normalised probabilities
			normalised_probabilities = probabilities / probabilities.sum()

			#New Topic for the current word
			new_topic = np.random.multinomial(1,normalised_probabilities).argmax()

			#Create updates 
			doc_word_topic[counter][word_counter] = new_topic

			theta[counter,new_topic] += 1

			phi[new_topic,word] += 1

			word_count_topic[new_topic] += 1

			
	return 


#Function to figure out the most frequent words for each topic 
def topic_words(docs, vocab, doc_word_topic, K):
	#Initialise dictionary
	count_dict = []
	for i in range(K):
		count_dict.append(dict())

	#Iterating through matrix containing document-word-topic information
	for i, doc in enumerate(doc_word_topic):
		for j, topic in enumerate(doc):
			#Key already present - Update!
			if docs[i][j] in count_dict[topic]:
				count_dict[topic][docs[i][j]] += 1
			else:
				count_dict[topic][docs[i][j]] = 1

	#Sort the word-topic-counts
	sorted_counts = [sorted(topic.items(),key=operator.itemgetter(1),reverse=True) for topic in count_dict]

	topic_top_words = []
	#Replace positions with word
	for i in range(K):
		topic_words = sorted_counts[i]
		temp = []
		for j in range(len(topic_words)):
			temp.append(vocab[topic_words[j][0]])

		topic_top_words.append(temp)


	return topic_top_words

#Learning algorithm 
def learn_LDA(docs,vocab,K,alpha,beta,iterations):
	#Cardinality of vocabulary 
	V = len(vocab)

	### Initialise working matrices and vectors ###

	#Matrice containing topic count for each word in the document --- Shape similar to document matrix
	doc_word_topic = []

	#Matrice containing document - topic distribution : Initialised with dirichlet prior alpha
	theta = np.zeros((len(docs),K)) + alpha

	#Matrice containing topic - word distributions : Initialised with dirichlet prior beta 
	phi = np.zeros((K,V)) + beta

	#Matrice containing total word count for each topic 
	word_count_topic = np.zeros(K) + V*beta 

	#Initialise all the matrices before learning / inference 
	initialise_matrices(docs, doc_word_topic,theta,phi,word_count_topic)

	#Initial Perplexity of the model
	initial_perplexity = compute_perplexity(alpha,docs, doc_word_topic,theta,phi,word_count_topic)

	#Perform iterations until stabilisation
	for i in range(iterations):
		#print(theta)
		infer_learn(docs, doc_word_topic,theta,phi,word_count_topic)
		#print(theta)
		new_perplexity = compute_perplexity(alpha,docs, doc_word_topic,theta,phi,word_count_topic)

		#Exit if perplexity increases
		if initial_perplexity < new_perplexity:
			break
			

	#Find most frequent words for each topic 
	final_word_topic_list = topic_words(docs, vocab, doc_word_topic,K)

	return 


#Initialising model parameters
def model(docs, vocab):
	###### Parameters #####

	#Number of Topics
	K = 2

	#Dirichlet prior on the per document topic distribution
	alpha = 0.5

	#Dirichlet prior on the per topic word distribution
	beta = 0.1 

	#Number of iterations
	iterations = 100

	learn_LDA(docs,vocab,K,alpha,beta,iterations)

	return 

######################################### Preprocessing Step for the documents ########################################

#Function to create a small corpus of documents
def get_test_docs():
	doc_a = "Brocolli is good to eat My brother likes to eat good brocolli but not my mother"
	doc_b = "My mother spends a lot of time driving my brother around to baseball practice"
	doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure"
	doc_d = "I often feel pressure to perform well at school but my mother never seems to drive my brother to do better"
	doc_e = "Health professionals say that brocolli is good for your health"

	final_docs = [doc_a, doc_b, doc_c, doc_d, doc_e]

	return final_docs

#Function to clean the documents into usable form
def clean_docs(docs):
	#Convert into lower case
	docs = [doc.lower() for doc in docs]

	final_docs = [doc.split() for doc in docs]

	return final_docs

#Function to create a vocabulary list 
def create_vocab(docs):
	sing_list = [item for sublist in docs for item in sublist]

	vocab = list(set(sing_list))

	#Create a new doc matrix - with term numbers
	new_doc = []

	for ele in docs:
		temp = []
		for word in ele:
			temp.append(vocab.index(word))

		new_doc.append(temp)

	return vocab, new_doc

#Main function for subsidiary calls
def main():
	#Test documents 
	docs = get_test_docs()

	#Cleaning Step for the sentences
	cleaned_docs = clean_docs(docs)

	#Creation of vocabulary / corpus of words
	vocabulary, docs = create_vocab(cleaned_docs)

	#Create and learn the LDA model
	model(docs, vocabulary)

	return


main()


#####################################################################################################################################

