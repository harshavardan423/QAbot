import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def categorize_sentences(text):
    sentences = text.split(".")
    categorized_sentences = {
        'introduction': [],
        'context': [],
        'suggestions': [],
        'summary': []
    }
    category_rules = {
    'introduction': ['introduction', 'background info',
                     "what is","who is","why is"
                     "Firstly", "In this article", 
                     "This paper will discuss", 
                     "This essay will explore", 
                     "To begin with","is","is the"], 

    'context': ['history', 
                'current situation', 
                'additional facts',
                'In addition',
                "In order to understand", 
                "The purpose of this section is", 
                "To provide background", 
                "To illustrate the point", 
                "To provide context", ], 

    'suggestions': ['recommendations', 
                    'next steps','suggestions',
                    "It is suggested that", 
                    "It could be argued that", 
                    "It is recommended that", 
                    "It would be beneficial to", 
                    "good reasons",
                    "It is proposed that","you can","you can also","you should try","So what should you do?","helps","useful","to avoid","In addition"],

    'summary': ['key takeaways','In summary','takeaway',"In conclusion", 
                "To summarise", 
                "To summarise the main points", 
                "In summary", 
                "To recap",],
    'action': ['Do this', 'Do that', 'Do that too']
}

    current_categories = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if any(keyword in sentence.lower() for keyword in category_rules['introduction']):
            current_categories = ['introduction']
        elif any(keyword in sentence.lower() for keyword in category_rules['context']):
            current_categories = ['context']
        elif any(keyword in sentence.lower() for keyword in category_rules['suggestions']):
            current_categories = ['suggestions']
        elif any(keyword in sentence.lower() for keyword in category_rules['summary']):
            current_categories = ['summary']
        else:
            current_categories = current_categories

        for category in current_categories:
            categorized_sentences[category].append(sentence)

    return categorized_sentences



text = """ 
Why is exercise important for older people? Getting your heart rate up and challenging your muscles benefits virtually every system in your body and improves your physical and mental health in myriad ways. Physical activity helps maintain a healthy blood pressure, keeps harmful plaque from building up in your arteries, reduces inflammation, improves blood sugar levels, strengthens bones, and helps stave off depression. In addition, a regular exercise program can make your sex life better, lead to better quality sleep, reduce your risk of some cancers, and is linked to longer life.
"""
categorized_sentences = categorize_sentences(text)
print(categorized_sentences)
