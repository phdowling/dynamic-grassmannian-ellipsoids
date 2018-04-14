import string

from nltk import TreebankWordTokenizer

stopwords = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as",
             "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
             "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
             "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
             "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
             "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
             "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
             "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
             "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
             "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
             "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
             "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
             "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
             "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"}
#             "!", "...", ",", ".", "?", "-", "``", "''", "{", "}", "[", "]", "(", ")", "’", ":", ";",
#             "--"}

translator = str.maketrans('', '', string.punctuation)


def clean_sent(sent, lower=True, filter_stopwords=True):
    if filter_stopwords:
        _stopwords = stopwords
    else:
        _stopwords = {}
    # sent = sent.translate(translator).strip().split()
    if lower:
        sent = sent.strip().lower().split()
        sent = [word for word in sent if word not in _stopwords]
    else:
        sent = sent.strip().split()
        sent = [word for word in sent if word.lower() not in _stopwords]
    return " ".join(sent)


tok = TreebankWordTokenizer()


def tokenize(t, filter_stopwords=True):
    return tok.tokenize(clean_sent(t, filter_stopwords=filter_stopwords))
