#import fileinput
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import re
from collections import defaultdict


def download_punkt():                                                           #Download the nlkt tokenizer data if missing
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


class CreateVocabulary:                                                         #Create 2 dictionaries mapping all unique words in the working text. For word and index lookup respectivily   
    def __init__(self, word_list):
        self.word2indx_list = {}
        self.indx2word_list = {}
        self.build_vocab(word_list)

    def build_vocab(self, word_list):
        idx = 0        
        for word in word_list:
            if word not in self.word2indx_list and re.match("^[a-zA-Z]+$", word):           #Do not include duplicates or non-aplhabetical characters  Note: Include numbers?
                self.word2indx_list[word] = idx
                self.indx2word_list[idx] = word
                idx += 1

    def word2indx(self, word):
        return self.word2indx_list.get(word, None)                              #Return None if word not found
    
    def indx2word(self, indx):
        return self.indx2word_list.get(indx, -1)                                #Return -1 if index not found ("None" could be a existing word)
    
class PairFrequencyTable():                                                     #Creates frequency table of pair of words that are n steps/words seperated in text. Note: It takes 1 step to pass first/original word. Meaning 2 steps will lead to the word next to the first word
    def __init__(self, word_list, n_step):
        self.table = defaultdict(lambda: defaultdict(int))                      #Creates dict where the default value of any new key is a new dict initilized with value 0 (the frequency of the word pair) The keys in both outer and inner dicts consist of words
        self.n = n_step                                                         #Variable to identify what kind of table it is. ex: n = 3, Frequency table of 3 steps
        self.create_pair_frequency_table(word_list, n_step)

    def create_pair_frequency_table(self, word_list, n_step):                   #Initilizes frequency table by generating ngrams and storing the first and last word in each tuple 
        ngram_list = ngrams(word_list, n_step)
        for ngram_tuple in ngram_list:
            first_word, last_word = ngram_tuple[0], ngram_tuple[-1]             #[-1] "shortcut" for accessing last element in list
            self.table[first_word][last_word] += 1

    def print_table(self):
        with open("pair_frequensies.txt", 'w') as file:
            for first_word, following_words in self.table.items():
                sorted_following_words = dict(sorted(following_words.items(), key=lambda item: item[1], reverse=True))
                file.write(f"{first_word}: {dict(sorted_following_words)}\n")

def create_frequency_table(word_list, vocabulary):
    frequency_table = defaultdict(int)
    for word in word_list:
        index = vocabulary.word2indx(word)
        if index is not None:
            frequency_table[index] += 1
    return dict(frequency_table)



#------------------------------MAIN------------------------------

def main():
    try:
        with open("sample.txt") as file:                            #Open text file and input content into "text" variable
            text = file.read()
    except FileNotFoundError:
        print("File 'Sample.txt' not found")
        return
    except IOError:
        print("Error reading the file")
        return

    word_list = word_tokenize(text)                                 #Split text into words/tokens with help from nltk built in models. (Will also tokenize symbols like ., [, &)
    vocabulary = CreateVocabulary(word_list)

    
    wordfreq = create_frequency_table(word_list, vocabulary)
    pairfreq = PairFrequencyTable(word_list, 4)

    pairfreq.print_table()
    """ print("Frequency Table:")
    for indx, freq in wordfreq.items():
        word = vocabulary.indx2word(indx)
        print(f"{word} ({indx}): {freq}")
    """


if __name__ == "__main__":
    download_punkt()
    main()