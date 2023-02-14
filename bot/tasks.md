## assign question

    # Read the JSON file
    with open("qa.json", "r") as json_file:
        data = json.load(json_file)

    # Extract the questions and answers
    questions = data["questions"]
    answers = data["answers"]


## take input

## check which tag is getting matched 
## check tag in qa.json
    ## get tag patterns as keywords
## assign score for order accuracy 

    inp.split

    for word in words:
        if word not in dictionary:
            return 0
        if prev_word != "":
            score += 1 if dictionary.index(prev_word) < dictionary.index(word) else 0
        prev_word = word
    return score