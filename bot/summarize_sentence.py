import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

def summarize_text(text, num_sentences=3):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.isalnum() and w not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    # Calculate word frequencies
    freq_dist = nltk.FreqDist(words)

    # Sort sentences by total word frequency
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        words_in_sentence = word_tokenize(sentence.lower())
        words_in_sentence = [w for w in words_in_sentence if w.isalnum() and w not in stop_words]
        sentence_score = sum([freq_dist[w] for w in words_in_sentence])
        sentence_scores[i] = sentence_score

    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

    # Get the top N sentences and sort them in order of appearance in the text
    top_sentences = sorted(sorted_sentences[:num_sentences], key=lambda x: x[0])

    # Return the summarized text
    summary = ' '.join([sentences[i] for i, _ in top_sentences])
    return summary


text = """

"""
summary = summarize_text(text, num_sentences=2)
print(summary)
