#import fileinput
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.util import ngrams
import numpy as np
import re
from collections import defaultdict
import random
from pprint import pprint                                                       #Debug aid


def download_punkt():                                                           #Download the nlkt tokenizer data if missing
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')


#Incorporate POS-tags in existing word dict or create new dict variable? (New variable storing over 20K entries but easier code management and debugging)
class CreateVocabulary:                                                             #Create 2 dictionaries mapping all unique words in the working text. For word and index lookup respectivily   
    def __init__(self, pos_tags):
        self.word2indx_list = {}
        self.indx2word_list = {}
        self.build_vocab(pos_tags)

    def build_vocab(self, pos_tags):
        idx = 0        
        for word, tag in pos_tags:
            if word not in self.word2indx_list:                                     #Do not include duplicates
                self.word2indx_list[word] = (idx, [tag])                            #Store index and POS tag as list (there can be multiple tags)
                self.indx2word_list[idx] = word
                idx += 1
            else:                                                                   #If word exist only append the POS tag
                existing_index, existing_tags = self.word2indx_list[word]
                if tag not in existing_tags:                                        #Only append if it is a new tag
                    existing_tags.append(tag)
                    self.word2indx_list[word] = (existing_index, existing_tags)

    def word2indx(self, word):
        word = self.word2indx_list.get(word, None)
        return word[0] if word else None                                            #Return only word index. First element in the tuple
    
    def indx2word(self, indx):
        return self.indx2word_list.get(indx, -1)                                    #Return -1 if index not found ("None" could be a existing word)

    def word_pos_tag(self, word):
        return self.word2indx_list.get(word, (None, None))[1]

class PairFrequencyTable():                                                         #Creates frequency table for pair of words that are n words long. Note: First word is n=1, meaning lowest possible value for a pair is n=2 
    def __init__(self, word_list, n_step = 2):
        self.table = defaultdict(defaultdict(int).copy)                             #Key is the first word. Value is another dict where the key is the word pair and value it's frequency. [Word]-[WordPair]:[Frequency]
        #self.n = n_step                                                            #Variable to identify what kind of table it is. ex: n = 3, Frequency table of 3 steps
        self.create_pair_frequency_table(word_list, n_step)

    def create_pair_frequency_table(self, word_list, n_step):                       #Initilizes frequency table by generating ngrams and storing the first and last word in each tuple as a pair
        ngram_list = ngrams(word_list, n_step)
        for ngram_tuple in ngram_list:
            first_word, last_word = ngram_tuple[0], ngram_tuple[-1]                 #[-1] will access last element in list
            self.table[first_word][last_word] += 1

    def print_table(self, iteration = 2):
        filename = f"pair_frequensies_{iteration}.txt"
        with open(filename, 'w') as file:
            for first_word, following_words in self.table.items():
                sorted_following_words = dict(sorted(following_words.items(), key=lambda item: item[1], reverse=True))  #Sort the inner dict in terms of value(frequency) with the highest frequency first
                file.write(f"{first_word}: {dict(sorted_following_words)}\n")


#Stand alone table for the frequency of each seperate word 
def create_frequency_table(word_list, vocabulary):
    frequency_table = defaultdict(int)
    for word in word_list:
        index = vocabulary.word2indx(word)
        if index is not None:
            frequency_table[index] += 1
    frequency_table = sorted(frequency_table.items(), key=lambda item:item[1], reverse=True)
    return dict(frequency_table)

def print_frequency_table(wordfreq, vocabulary):
    with open("frequency_table", 'w') as file:
        for word, freq in wordfreq.items():
            file.write(f"{vocabulary.indx2word(word)}: {freq}\n")
            

#Main implementation for text generation 
class MarkovChain:
    def __init__(self, pairfreq_tables, vocabulary):
        self.pairfreq_tables = pairfreq_tables
        self.vocabulary = vocabulary
        self.transition_matrices = self.build_transition_matrix()

    def build_transition_matrix(self):
        vocab_size = len(self.vocabulary.word2indx_list)                                        #Amount of unique words in the vocabulary
        transition_matrices = []

        #Initilize the matrixes and store them in a list 
        for _ in range(10):
            transition_matrix = np.zeros((vocab_size, vocab_size))                              #Create a matrix filled with zeros (2D NumPy array) 
            transition_matrices.append(transition_matrix)

        #Loop through each word and it's index, in the vocabulary 
        for word, (index, _) in self.vocabulary.word2indx_list.items():                          
            
            for n in range(10):
                following_words = self.pairfreq_tables[n].table.get(word, None)                     #Get all pair-words for the current word
            
                if following_words:
                    total = sum(following_words.values())                                           #Calculate the total frequency of all pair-words/following words to the current word
                    for next_word, count in following_words.items():                                #Loop through each pair-word and it's frequency 
                        next_index = self.vocabulary.word2indx(next_word)                           #Get the index of the current pair-word (For storage/look-up optimization)
                        transition_matrices[n][index][next_index] = count / total                   #Store the current word (Row) and current pair-word (Column), and the probability to transition to the pair-word

        return transition_matrices

    def build_POS_transition_matrix(self, pos_tags):
        pos_only = [tag for word, tag in pos_tags]                                                  #Only extract the POS tags
        unique_tags = list(set(pos_only))                                                           #Convert to set to remove duplicates and convert back to list again

        #Map/index the tags to aid the construction and look-up of the Numpy matrix
        pos_to_indx = {tag: i for i, tag in enumerate(unique_tags)}                                 #Dict where each tag has a numerical index 
        indx_to_pos = {i: tag for tag, i in pos_to_indx.items()}

        #Initialize matrix
        tag_amount = len(unique_tags)
        pos_transition_matrix = np.zeros((tag_amount, tag_amount))

        for i in range(len(pos_only)):
            current_tag = pos_only[i]                                                               #Row
            next_tag = pos_only[i+1]                                                                #Column
            pos_transition_matrix[ pos_to_indx[current_tag], pos_to_indx[next_tag] ] += 1           #Add 1 in intersection cell, corresponding to how often a tag follows another

        #Normilize counts to get probability
        row_sums = pos_transition_matrix.sum(axis=1)                                                #Returns an array with the sum of each rown in the matrix
        pos_transition_matrix = pos_transition_matrix / row_sums[:, np.newaxis]                     #row_sums is a 1D array, operation to change it's shape to 2D to be compatible for division with 2D matrix

    def generate_sentence(self, start_word=None, length=10):
        
        if not start_word:                                                                  #Default start word when no input
            start_word = self.vocabulary.word2indx("call")

        current_word = self.vocabulary.indx2word(start_word)
        sentence = [current_word]
        
        #Generate sentence word for word 
        for _ in range(length-1):
            current_word = self.get_next_word(sentence)

            if not current_word:
                break
            sentence.append(current_word)
           
        
        sentence_str = ' '.join(sentence)                                                     #Format list containing the sentence to string. For print functionality 
        print(sentence_str)
        return sentence_str                                                                   #Return sentence (currently unused)

    def get_next_word(self, sentence):
        last_word_index = self.vocabulary.word2indx(sentence[-1])
        probabilities = self.transition_matrices[0][last_word_index]                          #Returns the row of probabilites for word pairs corresponding to the last word in the sentence 

        #Loop through all previous words in the sentence
        for n, previous_words in enumerate(reversed(sentence[:-1])):                          #Sentence[:-1] omits the last word since that is already handled in previous line of code
            if n > 10:                                                                        #Max size for sentence is 10 words. Can be removed, sentence should not extend set length
                break                             
        
            previous_word_index = self.vocabulary.word2indx(previous_words)

            #Multiply the probabilties of the potential words being n-step back the current word in the sentence 
            values = self.transition_matrices[n+1][previous_word_index]                       #NOTE: If there is a 0 frequency this might set the whole probability to 0 for that word. Need further investigation
            
            for i, value in enumerate(values):                                                #Work around for when a value in the matrix is 0
                if value > 0:
                    probabilities[i] *= value                                                 #Future note: Due to the sturcture of the matrices, this should multiply the values of the same words. Look into if this is actually correct
                                                                                              #Future future note: This should be correct. The columns should be close to identical for the previous n-grams. Making i and value be the same word in this context
        #Cases when there is no next word to transition to
        if np.sum(probabilities) == 0:                                                      
            return None
        
        #Create probability distribution by normalizing counts 
        probabilities_sum = np.sum(probabilities)                                             #Matrix should already have normalized values. However, seems to be edge cases where that is not the case thus this code is necessary
        if probabilities_sum != 1:                                                            #Note: Finding a way to solve the problem and thus removing this code. Could yield better performance  
            probabilities = probabilities / probabilities_sum
        
        next_index = np.random.choice(len(probabilities), p = probabilities)                  #Get next word, weighed on the calculated probabilites 
        return self.vocabulary.indx2word(next_index)


#------------------------------------------------MAIN------------------------------------------------

def main():
    file_names = ["sample.txt", "A_room_with_a_view.txt"]
    all_text = ""

    for file_name in file_names:
        try:
            with open(file_name) as file:                            #Open text files and input content into "text" variable
                text = file.read()
                all_text += text
        except FileNotFoundError:
            print(f"File '{file_name}' not found")
            return
        except IOError:
            print(f"Error reading the file '{file_name}'")
            return

     
    tokenizer = TweetTokenizer(preserve_case = False)
    corpus = re.sub(r"[‘’´`]", "'", all_text)                                                                                              #Edge case where apostrophes can have different Unicode. This normalize them
    word_list = tokenizer.tokenize(corpus)                                                                                                  #Split text into words/tokens with help from nltk built in models. ( Will also tokenize symbols e.g ., [, & )      
    word_list = [token for token in word_list if re.match(r"^[\w]+(?:['-][\w]+)*$", token) and "_" not in token]                           #Clean tokens by removing special characters (edge case for "_" had to be included since it was not considered a character)
    pos_tags = pos_tag(word_list)                                                                                                           #Part-of-speech tagging. NLTK function that tags each word with a grammatical tag (Noun, verb, etc)

    
    vocabulary = CreateVocabulary(pos_tags)                                                                                                #pos_tags contain all information that word_list has with the added bonus of grammatical tags
    print(len(word_list))
    wordfreq = create_frequency_table(word_list, vocabulary)        #Frequency of each word in the corpus 
    print_frequency_table(wordfreq, vocabulary)

    #List to store pairfrequency tables to aid in dynamic creation and handling of several tables
    pairfreq_tables = []                                            
    for n in range(2, 12):  
        pairfreq = PairFrequencyTable(word_list, n)
        pairfreq_tables.append(pairfreq)
        pairfreq.print_table(n)

        
    
    #Weighted random choice for initial word in text generation
    words = list(wordfreq.keys())
    weights = list(wordfreq.values())
    initial_word = random.choices(words, weights=weights)[0]

    #Text generation with implementation of MarkovChain algorithm  
    generated_text = MarkovChain(pairfreq_tables,vocabulary)
   
    for _ in range(1,5):
        generated_text.generate_sentence(initial_word)

if __name__ == "__main__":
    download_punkt()
    main()