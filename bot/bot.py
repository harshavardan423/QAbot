import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
stemmer = LancasterStemmer()
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



with open("C:/Users/H2/Documents/GitHub/QAbot/bot/qa_2.json", encoding="utf-8") as file:
    data = json.load(file)

# QA FILE
with open("C:/Users/H2/Documents/GitHub/QAbot/bot/qa_2.json",encoding="utf-8") as json_file:
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

# def learn_new_intent(intent, patterns, responses):
#     with open("intents.json", "r") as file:
#         data = json.load(file)
#     new_intent = {"tag": intent, "patterns": patterns.split(","), "responses": responses.split(",")}
#     data["intents"].append(new_intent)
#     with open("intents.json", "w") as file:
#         json.dump(data, file)

def chat():
    # intents_action = { "search" : search_web, "Identify" : search_web}
    print(Back.WHITE + Style.DIM +"Chat" + Back.RESET +  Style.RESET_ALL)
    
    
    while True:

       

        # for question in qa_data["questions"]:
            

        ## PRINTING THE QUESTION
        # print(Fore.GREEN + question["tag"] + Fore.RESET)
        # print()

        inp = input(Fore.GREEN +  "You: " + Fore.RESET )

        if inp.lower() == "quit":
            break

    


        # elif inp.split()[0] in intents_action :
        #     command = inp.split()[0]
        #     command_text = inp.replace(command, "", 1).strip() # remove the command and leading/trailing whitespaces
        #     if command_text: # check if the command text is not empty
        #         intents_action[command](command_text)
        #     else:
        #         print(f"Please provide a valid {command} text")
        
        # elif inp.lower().startswith("#learn") :
        #     try:
        #         intent, patterns, responses = inp.lower().replace("#learn", "", 1).strip().split(" ", 2)
        #         learn_new_intent(intent, patterns, responses)
        #         print("New Intent Learned!")
        #     except ValueError:
        #         print("Invalid Input format, the format should be #learn intent patterns responses")
        else:
            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]

            if results_index < len(labels):
                tag = labels[results_index]
                confidence = results[results_index]
            else:
                # tag = None
                confidence = 0
                continue

            # print("Tag : " + tag)
            # print()
        
            keywords = []

            
            
                
                    
            
            for tg in data["questions"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    
                    
            
                    f_response = random.choice(responses)

                    

                    if confidence > 0.95 :
                        # print1by1(Fore.RED + f"{f_response} \n" + Fore.RESET )
                        keywords = (dm_2.generate_sentences(f_response))
                        
                        score = scoring_system(str(inp),keywords)
                            
                        print(Fore.GREEN + "Similarity Scores" + Fore.RESET, score)
                        print("Confidence : "+ str(confidence))
                        print()
                        continue

                        
                    elif confidence < 0.95 :
                        score = 0
                        print1by1(Fore.RED + f".... \n recorded response \n" + Fore.RESET )
                        continue
                # if results_index >  0.1 :
                
                # if tag in intents_action:
                #     intents_action[tag](inp)

                

                # if score == 0.0 :
                # print1by1(Fore.YELLOW + f"{random.choice(responses)} \n" + Fore.RESET )
                # print1by1(Fore.YELLOW + f"{random.choice(context)} \n" + Fore.RESET )

                # if context :
                #     print1by1(Fore.YELLOW + f"{random.choice(context)} \n" + Fore.RESET )
        
                # print(tag)
                # print(results)

                # else :
                #     print1by1(Fore.BLUE + f"??? \n" + Fore.RESET )   


def scoring_system(words, dictionary):
    word_list = words.split()
    
    # s_score = fuzz.token_set_ratio(word_list,dictionary)
    matching_words = list(set(word_list).intersection(dictionary))

    matcher = difflib.SequenceMatcher(None, matching_words, dictionary)
    d_score = matcher.ratio()*100

    # j_score = len(set(word_list).intersection(dictionary)) / len(set(word_list).union(dictionary))
      
    return d_score
    


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