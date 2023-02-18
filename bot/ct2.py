import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define rules for the introduction category
introduction_patterns = [ 
    [{"LOWER": "introduction"}],
    [{"LOWER": "background"}, {"LOWER": "info"}],
    [{"LOWER": "about"}, {"POS": "NOUN"}],
    [{"LOWER": "what"}, {"LOWER": "is"}],
    [{"LOWER": "why"}, {"LOWER": "is"}],
    [{"LOWER": "to"}, {"LOWER": "begin"}],
    [{"LOWER": "to"}, {"LOWER": "start"}],
    [{"LOWER": "let's"}, {"LOWER": "start"}],
    [{"LOWER": "let's"}, {"LOWER": "begin"}],
]
matcher.add("INTRODUCTION", introduction_patterns)

# Define rules for the context category
context_patterns = [
    [{"LOWER": "history"}],
    [{"LOWER": "current"}, {"LOWER": "situation"}],
    [{"LOWER": "additional"}, {"LOWER": "facts"}],
    [{"LOWER": "in"}, {"LOWER": "this"}, {"LOWER": "context"}],
    [{"LOWER": "to"}, {"LOWER": "understand"}],
    [{"LOWER": "more"}, {"LOWER": "about"}],
    [{"LOWER": "in"}, {"LOWER": "relation"}, {"LOWER": "to"}],
    [{"LOWER": "in"}, {"LOWER": "terms"}, {"LOWER": "of"}],
]
matcher.add("CONTEXT", context_patterns)

# Define rules for the suggestions category
suggestions_patterns = [
    [{"LOWER": "recommendations"}],
    [{"LOWER": "next"}, {"LOWER": "steps"}],
    [{"LOWER": "suggestions"}],
    [{"LOWER": "ideas"}],
    [{"LOWER": "tips"}],
    [{"LOWER": "ways"}, {"LOWER": "to"}],
]
matcher.add("SUGGESTIONS", suggestions_patterns)

# Define rules for the summary category
summary_patterns = [
    [{"LOWER": "key"}, {"LOWER": "takeaways"}],
    [{"LOWER": "in"}, {"LOWER": "summary"}],
    [{"LOWER": "to"}, {"LOWER": "summarize"}],
    [{"LOWER": "conclusion"}],
    [{"LOWER": "in"}, {"LOWER": "conclusion"}],
    [{"LOWER": "to"}, {"LOWER": "conclude"}],
]
matcher.add("SUMMARY", summary_patterns)

# Define rules for the action category
action_patterns = [
    [{"LOWER": "do"}, {"LOWER": "this"}],
    [{"LOWER": "do"}, {"LOWER": "that"}],
    [{"LOWER": "do"}, {"LOWER": "that"}, {"LOWER": "too"}],
    [{"LOWER": "to"}, {"LOWER": "take"}],
    [{"LOWER": "take"}, {"LOWER": "action"}],
    [{"LOWER": "what"}, {"LOWER": "you"}, {"LOWER": "can"}, {"LOWER": "do"}],
]
matcher.add("ACTION", action_patterns)

# Define a function to perform the text categorization
def categorize_text(text):
    doc = nlp(text)
    categories = {"introduction": [], "context": [], "suggestions": [], "summary": [], "action": []}
    for match_id, start, end in matcher(doc):
        category = nlp.vocab.strings[match_id]
        categories[category].append(doc[start:end].text)
    return categories
text = "This is an introduction to the topic. It provides basic info about what it is and why it's important. In this context, it's useful to know some extra info about the history of the subject. Here are some suggestions for further reading on the topic.Do this,do that.DO that too. In summary, the main takeaway is that this is a complex and multifaceted subject."
categorized_sentences = categorize_text(text)
print(categorized_sentences)
