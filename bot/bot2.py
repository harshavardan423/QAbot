import nltk 
from nltk.stem.lancaster import LancasterStemmer
import json
stemmer = LancasterStemmer()
from gensim.models import KeyedVectors
import numpy as np
from difflib import SequenceMatcher
import spacy
import re
from transformers import AutoTokenizer, AutoModel
import torch
nlp = spacy.load("en_core_web_md")
import logging
import numpy
import pyfiglet
from colorama import Fore, Back, Style
from fuzzywuzzy import fuzz
import sys
import time
import difflib
import dm_2
import tflearn
import tensorflow
import random
import json
from googlesearch import search
import pickle
import os
from nltk.corpus import stopwords
from nltk import pos_tag
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


dev_mode = False
if dev_mode == False :
    tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.disable(logging.CRITICAL)


def pad_data(training, output):
        if len(training) > len(output):
            padding = [[0 for _ in range(len(training[0]))] for _ in range(len(training) - len(output))]
            output.extend(padding)
        elif len(output) > len(training):
            padding = [[0 for _ in range(len(output[0]))] for _ in range(len(output) - len(training))]
            training.extend(padding)
        return training, output

with open("qa_2.json", encoding="utf-8") as file:
    data = json.load(file)

# QA FILE
with open("qa_2.json",encoding="utf-8") as json_file:
    qa_data = json.load(json_file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["questions"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        
        # for context in intent["tag"]:
        #     words.extend(nltk.word_tokenize(context))
        #     docs_x.append(nltk.word_tokenize(context))

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)


    

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# training, output = pad_data(training, output)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
# net = tflearn.reshape(net, [-1,209,8])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

print(training.shape)
print(output.shape)
    
try:
    model.load("model.tflearn")
except:
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

def ask(message):
    inp = message
    results = model.predict([bag_of_words(inp, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    if results_index < len(labels):
        tag = labels[results_index]
        confidence = results[results_index]
    else:
        confidence = 0
        
    if confidence >= 0.9:
        for tg in data["questions"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        
        f_response = random.choice(responses)
        return f_response
    else:
        print('fail')
        return 'fail'



# def chat():
#     # intents_action = { "search" : search_web, "Identify" : search_web}
#     print(Back.WHITE + Style.DIM +"Chat" + Back.RESET +  Style.RESET_ALL)

#     while True:

#         inp =  input(Fore.GREEN +  "You: " + Fore.RESET )

#         if inp.lower() == "quit":
#             break
#         else:
#             results = model.predict([bag_of_words(inp, words)])[0]
#             results_index = numpy.argmax(results)
#             tag = labels[results_index]

#             if results_index < len(labels):
#                 tag = labels[results_index]
#                 confidence = results[results_index]
#             else:
#                 # tag = None
#                 confidence = 0
#                 continue

#             # print("Tag : " + tag)
#             # print()
        
#             keywords = []

#             for tg in data["questions"]:
#                 if tg['tag'] == tag:
#                     responses = tg['responses']
                    
                    
            
#                     f_response = random.choice(responses)

#                     if confidence > 0.95 :
#                         # print1by1(Fore.RED + f"{f_response} \n" + Fore.RESET )
#                         keywords = (dm_2.generate_sentences(f_response))
                        
#                         score = scoring_system(str(inp),keywords)
                            
#                         print(Fore.GREEN + "Similarity Scores" + Fore.RESET, score)
#                         print("Confidence : "+ str(confidence))
#                         return f_response
#                         continue

                        
#                     elif confidence < 0.95 :
#                         score = 0
#                         print1by1(Fore.RED + f".... \n recorded response \n" + Fore.RESET )
#                         continue
                

def scoring_system(words, dictionary):
    word_list = words.split()
    
    # s_score = fuzz.token_set_ratio(word_list,dictionary)
    matching_words = list(set(word_list).intersection(dictionary))

    matcher = difflib.SequenceMatcher(None, matching_words, dictionary)
    d_score = matcher.ratio()*100

    # j_score = len(set(word_list).intersection(dictionary)) / len(set(word_list).union(dictionary))
      
    return d_score
    

from fuzzywuzzy import fuzz

def score(words, response):
    if response != 'fail' :
        # Tokenize and preprocess words and response
        question_tokens = word_tokenize(words.lower())
        answer_tokens = word_tokenize(response.lower())

        # Remove stop words and lemmatize tokens
        stop_words = set(stopwords.words('english'))
        question_tokens = [WordNetLemmatizer().lemmatize(token) for token in question_tokens if token not in stop_words]
        answer_tokens = [WordNetLemmatizer().lemmatize(token) for token in answer_tokens if token not in stop_words]

        # Count matching keywords
        matching_keywords = set(question_tokens).intersection(set(answer_tokens))
        matching_keywords_fraction = f"{len(matching_keywords)}/{len(set(answer_tokens))}"
        matching_keywords_fraction_n = len(matching_keywords)/len(set(answer_tokens))


        # Count matching sentences and missing sentences
        question_sentences = sent_tokenize(words)
        answer_sentences = sent_tokenize(response)
        matching_sentences = 0
        for question_sentence in question_sentences:
            best_match_score = 0
            for answer_sentence in answer_sentences:
                match_score = fuzz.token_sort_ratio(question_sentence, answer_sentence)
                if match_score > best_match_score:
                    best_match_score = match_score
            if best_match_score > 75:  # Minimum match score to consider a match
                matching_sentences += 1
        matching_sentences_fraction = f"{matching_sentences}/{len(answer_sentences)}"
        matching_sentences_fraction_n = matching_sentences/len(answer_sentences)
        missing_sentences = set(answer_sentences).difference(set(question_sentences))
    
        # Find missing words
        missing_words = set(question_tokens).difference(set(answer_tokens))

        # Calculate grammar score using Part-of-Speech tagging
        question_pos = pos_tag(question_tokens)
        answer_pos = pos_tag(answer_tokens)
        matching_pos = set(question_pos).intersection(set(answer_pos))
        grammar_score = "{:.2f}".format(len(matching_pos) / len(answer_pos))

        # Calculate similarity score using fuzzy matching
        similarity = 0
        if len(question_sentences) > 0 and len(answer_sentences) > 0:
            similarity_scores = []
            for question_sentence in question_sentences:
                for answer_sentence in answer_sentences:
                    similarity_scores.append(fuzz.token_set_ratio(question_sentence, answer_sentence))
            similarity = "{:.2f}".format(sum(similarity_scores) / len(similarity_scores))

        from similarity import list_similarity

        sim = list_similarity(answer_sentences,question_sentences)

         # Generate a text prompt based on the matching fraction
        if matching_keywords_fraction_n < 0.25 and matching_sentences_fraction_n < 0.25:
            prompt = "It appears that the two sets of text are quite dissimilar. You may want to consider revising one or both sets of text."
        elif (matching_keywords_fraction_n > 0.5 and matching_keywords_fraction_n < 0.75) and (matching_sentences_fraction_n > 0.5 and matching_sentences_fraction_n < 0.75):
            prompt = "There is some overlap between the two sets of text, but there is still room for improvement. Consider revising one or both sets of text to increase their similarity."
        elif (matching_keywords_fraction_n > 0.75 and matching_keywords_fraction_n < 0.85) and (matching_sentences_fraction_n > 0.75 and matching_sentences_fraction_n < 0.85):
            prompt = "The two sets of text are quite similar. Consider revising one or both sets of text to increase their similarity further."
        elif (matching_keywords_fraction_n > 0.95 and matching_keywords_fraction_n < 1.0) and (matching_sentences_fraction_n > 0.95 and matching_sentences_fraction_n < 1.0) or sim == 1.0 :
            prompt = "The two sets of text are very similar."
        else:
            prompt = ""

        return matching_keywords_fraction, matching_sentences_fraction, grammar_score, list(missing_sentences), list(missing_words), similarity, sim , prompt

    

        # Tokenize and preprocess words and response
        word_tokens = word_tokenize(words.lower())
        response_tokens = word_tokenize(response.lower())

        # Remove stop words and lemmatize tokens
        stop_words = set(stopwords.words('english'))
        word_tokens = [WordNetLemmatizer().lemmatize(token) for token in word_tokens if token not in stop_words]
        response_tokens = [WordNetLemmatizer().lemmatize(token) for token in response_tokens if token not in stop_words]

        # Count matching keywords
        matching_keywords = set(word_tokens).intersection(set(response_tokens))
        matching_keywords_fraction = f"{len(matching_keywords)}/{len(set(response_tokens))}"

        # Count matching sentences
        word_sentences = sent_tokenize(words)
        response_sentences = sent_tokenize(response)
        matching_sentences = set(word_sentences).intersection(set(response_sentences))
        matching_sentences_fraction = f"{len(matching_sentences)}/{len(set(response_sentences))}"

        # Calculate grammar score using Part-of-Speech tagging
        word_pos = pos_tag(word_tokens)
        response_pos = pos_tag(response_tokens)
        matching_pos = set(word_pos).intersection(set(response_pos))
        grammar_score = len(matching_pos) / len(response_pos)

        # print("WORDS :"+ len(words), "RESPONSE : " + len(response))
        # print( matching_keywords_fraction, matching_sentences_fraction, grammar_score)
        return matching_keywords_fraction, matching_sentences_fraction, grammar_score

# print(score("Quantum mechanics is a branch of physics that deals with the behavior of matter and energy at the atomic and subatomic level.","Quantum mechanics is a branch of physics that deals with the behavior of matter and energy at the atomic and subatomic level. It is known for the wave-particle duality, which states that particles can exhibit properties of both waves and particles. The uncertainty principle states that certain properties of a particle cannot be known simultaneously with perfect accuracy. In quantum mechanics, a system is described by a wave function, also known as a quantum state, which is governed by the Schr\u00f6dinger equation."))



def similarity_score(set1, set2):
    count = 0
    for s1 in set1:
        found_similar_sentence = False
        for s2 in set2:
            similarity = SequenceMatcher(None, s1, s2).ratio()
            print(f"Similarity between '{s1}' and '{s2}': {similarity}")
            if similarity > 0.8:
                found_similar_sentence = True
                break
        if found_similar_sentence:
            count += 1
    score = count / len(set1)
    return score


def print1by1(text, delay=0.01):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print



def search_web(search_text):
    if search_text.replace(" ","") == "search" :
          print("Searching for?")
    else :
        #print("Search for :", search_text)
        
        for j in search(search_text, num_results=1):
            print(j.title)
    # Add your search function here

def move(text):
    print("Moving:",text)

ascii_art = pyfiglet.figlet_format("QA")
print(Fore.YELLOW + " " + ascii_art + " " + Fore.RESET)

# chat()
# ask("Electricity is the flow of electrical energy through a material, known as a conductor. It is characterized by the presence of electric charge, which can be measured by a unit called the Coulomb. The amount of energy that flows through a conductor is determined by the voltage, measured in Volts, and the current, measured in Amperes. The resistance of a material determines the opposition to the flow of current, and this is measured in Ohms.")
# score("A salt bath furnace is able to maintain a constant temperature because the salt bath acts as a heat storage medium.", "A salt bath furnace is able to maintain a constant temperature because the salt bath acts as a heat storage medium. The high heat capacity of the salt bath allows it to absorb and release large amounts of heat without significant changes in temperature. Additionally, the bath can be stirred or agitated to ensure even heating and temperature distribution throughout the bath. This helps to prevent hot spots and temperature fluctuations, allowing for consistent and controlled heating.")