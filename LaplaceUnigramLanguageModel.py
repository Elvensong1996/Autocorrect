import math, collections


class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.words = set() #V
    self.total = 0  #N
    self.wordcount = collections.defaultdict(lambda: 0) #count(w)
    self.train(corpus)


  def train(self, corpus):
    """ Takes a corpus and trains your language model.
      Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
      for datum in sentence.data:
        word = datum.word

        #add to vocabulary(V)
        self.words.add(word)

        #increase total token(N)
        self.total += 1

        #keep the count of each word(count(w))
        self.wordcount[word] = self.wordcount[word] + 1


  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    #Apply smoothed unigram probabilities here
    score = 0.0
    for token in sentence:
      count = self.wordcount[token]
      unigramProb = math.log(float(count + 1) / (self.total + len(self.words)))
      score += unigramProb
    return score
