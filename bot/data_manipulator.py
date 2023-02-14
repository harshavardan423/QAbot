import json

with open("C:/Users/H2/Documents/GitHub/QAbot/bot/science.json") as file:
    data = json.load(file)


def fun1():

        # read the JSON file
    with open('python.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for intent in data['intents']:
        intent["Index"] = "python-" + str(intent["Index"])

    # write the modified object back to the file
    with open('python.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)

def fun2(data):
    for intent in data['intents']:
        tag = intent['tag']
        intent['patterns'] = [tag]

    with open('output.json', 'w') as f:
        json.dump(data, f)


fun2(data)