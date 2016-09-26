import math, collections


class CustomLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramcount = collections.defaultdict(lambda: 0)
    self.biagramcount = collections.defaultdict(lambda: 0)
    self.triagramcount = collections.defaultdict(lambda: 0)
    self.totals = 0 # Number of tokens
    self.train(corpus)


  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    biagramlist = []
    triagramlist = []
    for sentence in corpus.corpus:
      for i in range(len(sentence.data)):
        word = sentence.data[i].word
        self.totals += 1

        # Add to unigram count
        self.unigramcount[word] = self.unigramcount[word] + 1

        # Add to biagram count
        if word != '</s>':
          biagramlist.append(word)
          biagramlist.append(sentence.data[i + 1].word)
          self.biagramcount[tuple(biagramlist)] = self.biagramcount[tuple(biagramlist)] + 1
          del biagramlist[:]

        # Add to triagram count
        if i + 2 < len(sentence):
          triagramlist.append(word)
          triagramlist.append(sentence.data[i + 1].word)
          triagramlist.append(sentence.data[i + 2].word)
          self.triagramcount[tuple(triagramlist)] = self.triagramcount[tuple(triagramlist)] + 1
          del triagramlist[:]


  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # Apply stupid back-off here
    score = 0.0

    for i in range(len(sentence)):
      # First try triagram
      if i >= 2:
        triagramlist = [sentence[i - 2], sentence[i - 1], sentence[i]]
        if self.triagramcount[tuple(triagramlist)] != 0:
          triagramProb = math.log(float(self.triagramcount[tuple(triagramlist)]) /
                           self.biagramcount[tuple([sentence[i - 2], sentence[i - 1]])])
          score += triagramProb
          continue

      # Back-off to biagram
      if i >= 1:
        biagramlist = [sentence[i - 1], sentence[i]]
        if self.biagramcount[tuple(biagramlist)] != 0:
          biagramProb = math.log(0.4) + math.log(float(self.biagramcount[tuple(biagramlist)]) /
                                 self.unigramcount[sentence[i - 1]])
          score += biagramProb

        # If all else fail back-off to unigram
        else:
          unigramProb = math.log(0.4) + math.log(float(self.unigramcount[sentence[i]] + 1) /
                                           (self.totals + len(self.unigramcount)))
          score += unigramProb

      else: #if i = 0
        score += math.log(0.4) + math.log(float(self.unigramcount[sentence[i]] + 1) /
                                           (self.totals + len(self.unigramcount)))

    return score
