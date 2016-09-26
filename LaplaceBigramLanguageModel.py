import math, collections


class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.words = set() #V
    self.wordcount = collections.defaultdict(lambda: 0)
    self.biagramCounts = collections.defaultdict(lambda: 0) #c(w(n - 1), w(n))
    self. train(corpus)


  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """
    biagramlist = []
    for sentence in corpus.corpus:
      for i in range(len(sentence.data)):
        word = sentence.data[i].word

        # add to vocabulary(V)
        self.words.add(word)

        # keep the count of each word
        self.wordcount[word] = self.wordcount[word] + 1

        # add to biagram count
        if word != '</s>':
          biagramlist.append(word)
          biagramlist.append(sentence.data[i + 1].word)

          # Convert to tuple
          biagramtuple = tuple(biagramlist)
          self.biagramCounts[biagramtuple] = self.biagramCounts[biagramtuple] + 1

          #clear list
          del biagramlist[:]


  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # Apply laplace biagram here
    score = 0.0
    for i in range(len(sentence) - 1):
      biagramword = [sentence[i], sentence[i + 1]]
      biagramProb = math.log(float(self.biagramCounts[tuple(biagramword)] + 1) /
                             (self.wordcount[sentence[i]] + len(self.words)))
      score += biagramProb

    return score
