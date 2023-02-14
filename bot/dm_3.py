## THIS FILE MAKES SURE THE JSONS ARE IN THE RIGHT FORMAT FOR PROCESSING

import json

# read the intents_1.json file and load the content into a variable
with open("C:/Users/H2/Documents/GitHub/QAbot/bot/qa_2.json", "r",encoding='utf-8') as file:
    intents = json.load(file)

# loop through each intent in the intents list
# for intent in intents["intents"]:
#     # check if the pattern is a list or not
#     if not isinstance(intent["patterns"], list):
#         # if the pattern is not a list, convert it into a list with a single element
#         intent["patterns"] = [intent["patterns"]]

#     if not isinstance(intent["responses"], list):
#         # if the pattern is not a list, convert it into a list with a single element
#         intent["responses"] = [intent["responses"]]
        

for intent in intents["questions"]:
    # check if the pattern is a list or not
    if not (intent.get('subject') is None):
        print("value is present for given JSON key")
        # intent['keywords'] = intent['keywords']

# write the updated intents list back to the intents_1.json file
with open("C:/Users/H2/Documents/GitHub/QAbot/bot/qa_2.json", "w",encoding='utf-8') as file:
    json.dump(intents, file, indent=4)
