import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

def summarize_text(text, num_sentences=10):
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

Washington Capitals forward Garnet Hathaway kept a blog throughout the 2023 Navy Federal Credit Union NHL Stadium Series. The Capitals lost to the Carolina Hurricanes 4-1 at Carter-Finley Stadium in Raleigh, North Carolina, on Saturday. In his final entry, Hathaway discusses what went wrong for Washington, the experience of his first NHL outdoor game and how the team needs to move on quickly.

We talked about it as a team beforehand, how a win tonight would have made this game a great memory. To lose it, it stings. 

It stings more than ever because those were two points that we really needed right now; I think that's what guys will kind of dwell on, but we can't do anything else tonight about it. When you lose a game, you want to figure out immediately, what you could've done to win it and there are, unfortunately, a lot of things right now. 

It was one of those things where we wish we could have it back. It would make this moment a lot sweeter, but unfortunately, we can't. For tonight, we have to continue to enjoy this moment.

Everything leading up to the game was awesome. Those fans were rowdy. Even playing two-touch soccer in front of the fans -- a lot of Capitals fans traveled too -- and it was such a cool environment that I felt really fortunate to play in. That walkout before the game, it was special, along with the flyover, the fireworks, and having 56, 57,000 people in the stadium. And when the lights turned off and the cell phones turned on, that's a moment I'll never forget.

Giving up that first goal 2:11 into the game, that's not a recipe for success. More times than not, you're going to lose when you do that and you're trying to get out of that hole. And the earlier it happens in a game, the harder it is to get out of it and, unfortunately, they got the next one and they got the next one and they got the next one.

We have a few things that we look over at the end of the game and say, "Did we do this? Did it help us succeed?" I thought we put a lot of pressure on them in the first period and that goal hurt on the scoreboard, but I don't think it slowed us down mentally. We've been down one goal before. We've been down two and we've come back. There was never a moment in the game when we thought we couldn't come back from this.

That's the belief on this team and that's the leadership in this locker room. In order to come back, we've got to do a lot more things right.

"""
summary = summarize_text(text, num_sentences=2)
print(summary)
