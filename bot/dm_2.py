import json
from textblob import TextBlob
from textblob import Word
from textblob.wordnet import Synset
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords 

stop_words = set(stopwords.words('english'))



def generate_words_list(input_sentence):
    words = input_sentence.split()
    new_sentences = []
    filtered_sentence = [w for w in words if not w.lower() in stop_words]

    words = filtered_sentence
    return list(words)
    
    # for word in words:
    #     synonyms = []
    #     for syn in Word(word).get_synsets():
    #         for lemma in syn.lemmas():
    #             synonyms.append(lemma.name())
    #     for syn in synonyms:
    #         new_sentence = input_sentence.replace(word, syn)
    #         new_sentences.append(new_sentence)
    # return new_sentences

# def update_json_file(file_path):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#         for intent in (data['intents']):
           
#             for pattern in tqdm(intent['patterns']):
#                 generated_sentences = generate_words_list(pattern)
#                 intent['patterns'] = [pattern] + generated_sentences
                
                
#     with open(file_path, 'w') as f:
#         json.dump(data, f, indent=2)

# file_path = 'intents.json'  
# update_json_file(file_path)
