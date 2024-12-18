import os
from save_file_utils import save_variables, load_variables
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.util import ngrams
from nltk.corpus import webtext
from nltk.corpus import gutenberg
from nltk.corpus import reuters
import numpy as np
import re
from collections import defaultdict
import random
from pprint import pprint                                                       #Debug aid
import time
import sys
import threading

#---------------------------------OS and Data/File handling START---------------------------------

def download_punkt():                                                           #Download the nlkt tokenizer data if missing
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find("corpora/webtext")
        #nltk.data.find("corpora/gutenberg")
        #nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download("webtext")
        #nltk.download("gutenberg")
        #nltk.download("reuters")

def prompt_for_corpus():
    # Ask user if they want to use the previous corpus
    user_input = input("Do you wish to run with the previous corpus? [y/n]: ").strip().lower()
    
    if user_input == "n":
        # Delete the saved data file if it exists
        filepath = "saved_data.pkl"
        if os.path.exists(filepath):
            os.remove(filepath)
            print("Previous corpus deleted.")
        else:
            print("No saved corpus found to delete.")
    elif user_input != "y":
        print("Invalid input. Running with previous corpus by default.")

animation_running = True

def show_processing_animation():
    animation = "Processing corpus"
    dots = ""
    
    while animation_running:
        sys.stdout.write(f"\r{animation}{dots}")
        sys.stdout.flush()
        dots += "."
        if len(dots) > 3:
            dots = ""
        time.sleep(1)

def start_animation():
    global animation_running
    animation_running = True
    # Start the animation in a separate thread
    threading.Thread(target=show_processing_animation, daemon=True).start()

def stop_animation():
    global animation_running
    animation_running = False
    # Adding a final message to stop the animation gracefully
    sys.stdout.write("\rProcessing completed.            \n")
    sys.stdout.flush()

#---------------------------------OS and Data/File handling END---------------------------------


#Incorporate POS-tags in existing word dict or create new dict variable? (New variable storing over 20K entries but easier code management and debugging)
class CreateVocabulary:                                                                     #Create 2 dictionaries mapping all unique words in the working text. For word and index lookup respectivily   
    def __init__(self, pos_tags):
        self.word2indx_list = {}
        self.indx2word_list = {}
        self.build_vocab(pos_tags)

    def build_vocab(self, pos_tags):
        idx = 0        
        for word, tag in pos_tags:
            if word not in self.word2indx_list:                                             #Do not include duplicates
                self.word2indx_list[word] = (idx, [tag])                                    #Store index and POS tag as list (there can be multiple tags)
                self.indx2word_list[idx] = word
                idx += 1
            else:                                                                           #If word exist only append the POS tag
                existing_index, existing_tags = self.word2indx_list[word]
                if tag not in existing_tags:                                                #Only append if it is a new tag
                    existing_tags.append(tag)
                    self.word2indx_list[word] = (existing_index, existing_tags)

    def word2indx(self, word):
        word = self.word2indx_list.get(word, None)
        return word[0] if word else None                                                    #Return only word index. First element in the tuple
    
    def indx2word(self, indx):
        return self.indx2word_list.get(indx, -1)                                            #Return -1 if index not found ("None" could be a existing word)

    def word_pos_tag(self, word):                                                           #Returns a tuple of POS tags corresponding to the word 
        return self.word2indx_list.get(word, (None, None))[1]
    
    #DEBUG: prints word2indx (pos tags) to readable file
    def write_word2indx_to_file(self, filename="word2indx_debug.txt"):
        with open(filename, "w") as file:
            file.write("self.word2indx_list = {\n")
            for word, (index, pos_tags) in self.word2indx_list.items():
                pos_tags_str = repr(pos_tags)  # Use repr() to show list format in string
                file.write(f"    '{word}': ({index}, {pos_tags_str}),\n")
            file.write("}\n")
        print(f"Debug information written to {filename}")


class PairFrequencyTable():                                                                 #Creates frequency table for pair of words that are n words long. Note: First word is n=1, meaning lowest possible value for a pair is n=2 
    def __init__(self, word_list, n_step = 2):
        self.table = defaultdict(defaultdict(int).copy)                                     #Key is the first word. Value is another dict where the key is the word pair and value it's frequency. [Word]-[WordPair]:[Frequency]
        #self.n = n_step                                                                    #Variable to identify what kind of table it is. ex: n = 3, Frequency table of 3 steps
        self.create_pair_frequency_table(word_list, n_step)

    def create_pair_frequency_table(self, word_list, n_step):                               #Initilizes frequency table by generating ngrams and storing the first and last word in each tuple as a pair
        ngram_list = ngrams(word_list, n_step)
        for ngram_tuple in ngram_list:
            first_word, last_word = ngram_tuple[0], ngram_tuple[-1]                         #[-1] will access last element in list
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
    def __init__(self, pairfreq_tables, vocabulary, pos_tags, wordfreq):
        self.pairfreq_tables = pairfreq_tables
        self.vocabulary = vocabulary
        self.wordfreq = wordfreq
        self.pos_to_indx = {}                                                                       #Arranged when building POS transition matrix. Used for navigating matrixes 
        self.indx_to_pos = {}                               
        self.current_pos_tag = ''
        self.word2word_matrices = self.build_wordpair_matrix()                                   #Transition matrices based on word pairs
        self.pos2pos_matrix = self.build_POS2POS_transition_matrix(pos_tags)                            #Transition matrix on the probability of next POS tag based on current POS tag
        self.pos2word_matrix = self.build_emission_matrix(pos_tags)
        self.viterbi_probs = {}
        self.viterbi_paths = {}

###Initialization of Markov Model

    def build_wordpair_matrix(self):
        vocab_size = len(self.vocabulary.word2indx_list)                                            #Amount of unique words in the vocabulary
        transition_matrices = []

        #Initilize the matrixes and store them in a list 
        for _ in range(10):
            transition_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)                                  #Create a matrix filled with zeros (2D NumPy array) 
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

    #Probability for a POS tag to transition to another tag (Row: POS, Column: POS)
    def build_POS2POS_transition_matrix(self, pos_tags):
        pos_only = [tag for word, tag in pos_tags]                                                  #Only extract the POS tags
        unique_tags = list(set(pos_only))                                                           #Convert to set to remove duplicates and convert back to list again

        #Map/index the tags to aid the construction and look-up of the Numpy matrix
        self.pos_to_indx = {tag: i for i, tag in enumerate(unique_tags)}                                 #Dict where each tag has a numerical index 
        self.indx_to_pos = {i: tag for tag, i in self.pos_to_indx.items()}

        #Initialize matrix
        tag_amount = len(unique_tags)
        pos_transition_matrix = np.zeros((tag_amount, tag_amount))

        for i in range(len(pos_only) - 1):                                                          #Last tag will have no "next_tag", therefore the -1
            current_tag = pos_only[i]                                                               #Row
            next_tag = pos_only[i+1]                                                                #Column
            pos_transition_matrix[ self.pos_to_indx[current_tag], self.pos_to_indx[next_tag] ] += 1           #Add 1 in intersection cell, corresponding to how often a tag follows another

        #Normilize counts to get probability
        row_sums = pos_transition_matrix.sum(axis=1, keepdims=True)                                 #Returns an array with the sum of each rown in the matrix. Keepdims retains the 2D format instead of reshaping the output to a 1D array in row_sums
        pos_transition_matrix = pos_transition_matrix / row_sums                                                   
        
        #trans_print = pos_transition_matrix[pos_to_indx['NNP'], pos_to_indx['NN']]                 #Debug purpose 
        #print(f"{trans_print}")

        return pos_transition_matrix
    
    
    #Probability for each word given a POS tag (Row: POS, Column: Word)
    def build_emission_matrix(self, pos_tags):
        vocab_size = len(self.vocabulary.word2indx_list)
        pos_only = [tag for word, tag in pos_tags]
        unique_tags = list(set(pos_only))
        #Use the class global variable instead? 
        #pos_to_indx = {tag: i for i, tag in enumerate(unique_tags)}                                 #Dict where each tag has a numerical index 
        indx_to_pos = {i: tag for tag, i in self.pos_to_indx.items()}

        emission_matrix = np.zeros((len(unique_tags), vocab_size))

        for word, tag in pos_tags:
            word_indx = self.vocabulary.word2indx(word)
            pos_indx = self.pos_to_indx[tag]
            emission_matrix[pos_indx][word_indx] =+ 1


        #Insert small value in 0s. Excluding this introduces filtering for words mismatching current POS tag 
        emission_matrix[emission_matrix == 0] = 0.00000001

        #Re-normalize rows after previous adjustments 
        row_sums = emission_matrix.sum(axis=1, keepdims=True)
        emission_matrix = emission_matrix/row_sums

        return emission_matrix

##### Start of Text Generation

    def viterbi_init(self, start_word=None):

        if not start_word:
            start_word = self.generate_starting_word(self.wordfreq)

        starting_word_indx = self.vocabulary.word2indx(start_word)

        if starting_word_indx is None:
            raise ValueError(f"Starting word '{start_word}' not found in vocabulary")
       
        initial_probs = {}

        for pos_tag in self.vocabulary.word_pos_tag(start_word):
            pos_index = self.pos_to_indx[pos_tag]

            emission_prob = self.pos2word_matrix[pos_index, starting_word_indx]
            initial_probs[pos_tag] = emission_prob

        #Normalize probabilities
        total_prob = sum(initial_probs.values())
        if total_prob == 0:
            raise ValueError(f"No valid emission probabilities for starting word '{start_word}'.")

        for pos_tag in initial_probs:
            initial_probs[pos_tag] /= total_prob

        #Init viterbi paths & probabilities
        self.viterbi_paths = {tag: [start_word] for tag in initial_probs}
        self.viterbi_probs = initial_probs

        return 
    
    #Input generated sentence to process optimal POS sequence and score of said sequence
    def viterbi_scoring(self, sentence, pos_sequence):
        
        self.viterbi_init(sentence[0])
        viterbi_POS_path = [None] * len(sentence)

        for i in range(1, len(sentence)):
            new_viterbi_probs = {}
            new_viterbi_paths = {}
            current_word = sentence[i]
            current_word_pos_tags = self.vocabulary.word_pos_tag(current_word)
            
            for current_tag in current_word_pos_tags:
                max_prob = 0
                best_prev_tag = None

                for prev_tag in self.viterbi_probs.keys():
                    
                    prev_prob = self.viterbi_probs[prev_tag]

                    #Prob of transitioning to current POS tag from previous one
                    pos_transition_prob = self.pos2pos_matrix[self.pos_to_indx[prev_tag], self.pos_to_indx[current_tag]]

                    #Prob of current word marked as current POS tag
                    emission_prob = self.pos2word_matrix[self.pos_to_indx[current_tag], self.vocabulary.word2indx(current_word)]

                    #
                    #Consider leveraging word pairs into the probability?
                    #

                    #Combine probabilities with previous steps
                    combined_prob = prev_prob * pos_transition_prob * emission_prob

                    if combined_prob > max_prob:
                        max_prob = combined_prob
                        best_prev_tag = prev_tag

                #Store path and probability 
                if max_prob > 0:
                    new_viterbi_probs[current_tag] = max_prob
                    new_viterbi_paths[current_tag] = self.viterbi_paths[best_prev_tag] + [current_word]                             #Update the path by appending current_word (value) in the list of words from the previous sequence (best_prev_tag)
                    
                    if len(viterbi_POS_path) < len(sentence):
                        viterbi_POS_path.append(current_tag)
                    else:
                        viterbi_POS_path[i-1] = current_tag

            self.viterbi_paths = new_viterbi_paths
            self.viterbi_probs = new_viterbi_probs

        #Find best sequence from the final tag with highest probability
        #best_final_tag = max(self.viterbi_probs, key=self.viterbi_probs.get)
        #best_sequence = self.viterbi_paths[best_final_tag]
        #score = max(self.viterbi_probs.values())

        #Make sure both list are of equal length 
        min_length = min(len(viterbi_POS_path), len(pos_sequence))
        viterbi_POS_path = viterbi_POS_path[:min_length]
        pos_sequence = pos_sequence[:min_length]

        #Compare and count amount of identical tags in same position
        matches = sum(1 for v_tag, p_tag in zip(viterbi_POS_path, pos_sequence) if v_tag == p_tag)

        #Count the amount of matches as percentage 
        score = (matches/min_length) * 100
        #print(viterbi_POS_path, '' ,score)

        return score

    #Hard-coded probabilities for the starting POS-tag in a sentence (Based on general observations of sentence structures, can be adjusted)
    def starting_pos_tag(self):
        start_pos_tags = ['DT', 'PRP', 'NN', 'NNP', 'JJ', 'CC', 'IN', 'RB', 'VB']
        start_probabilities = [0.25, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0.025, 0.025]

        start_tag = np.random.choice(start_pos_tags, p=start_probabilities)

        return start_tag
    
    #Find start word by only including words that match with starting POS tag and pick at random, with higher probability based on frequency 
    def generate_starting_word(self, wordfreq):
        self.current_pos_tag = self.starting_pos_tag()
        
        words_with_tag = []
        for word in wordfreq.keys():
            if self.current_pos_tag in self.vocabulary.word_pos_tag(self.vocabulary.indx2word(word)):
                words_with_tag.append(word)

        if not words_with_tag:                                                              #if no words with POS tag found. Fall back on frequency based pick
            words_with_tag = list(wordfreq.keys())
            print("Words with POS tag not found. Fall back on frequency based pick for start word")

        freq_weights = [wordfreq[word] for word in words_with_tag] 
        initial_word = random.choices(words_with_tag, weights=freq_weights)[0]

        return initial_word
        
    #The start and main function for generating text
    def generate_sentence(self, start_word=None, length=10):
        
        if not start_word:                                                                    #Default start word when no input
            start_word = self.generate_starting_word(self.wordfreq)

        current_word = self.vocabulary.indx2word(start_word)
        sentence = [current_word]
        pos_tag_sequence = []                                                                 #Debug: print purpose 
        pos_tag_sequence.append(self.current_pos_tag)
        
        #Generate sentence word for word 
        for _ in range(length-1):
            
            current_word = self.get_next_word(sentence)
            pos_tag_sequence.append(self.current_pos_tag)
            if current_word:              
                sentence.append(current_word)
           
        
        #pos_str = ' '.join(pos_tag_sequence)
        #print(pos_str)                                                                        #Debug: print the POS tags for the sentence
        
        #Returns score of how matching the POS sequence is to the computed POS sequence of the viterbi algorithm
        score = self.viterbi_scoring(sentence, pos_tag_sequence)
        
        return sentence, score                                                                       #Return sentence (currently unused)

    #Called in generate_sentence()
    def get_next_word(self, sentence):
        last_word_index = self.vocabulary.word2indx(sentence[-1])
        word_probabilities = self.word2word_matrices[0][last_word_index]                           #Returns the row of probabilites for word pairs corresponding to the last word in the sentence 

        #Get next POS tag
        current_pos_index = self.pos_to_indx[self.current_pos_tag]
        next_pos_probs = self.pos2pos_matrix[current_pos_index]
        #next_pos_index = np.random.choice(len(next_pos_probs), p=next_pos_probs)                    #Pick next pos tag weighed by the probabilities in pos_transition_matrix (Alt: pick POS with highest prob)
        next_pos_index = np.argmax(next_pos_probs)
        
        self.current_pos_tag = self.indx_to_pos[next_pos_index]                                     #Save the new POS tag        
        emission_probs = self.pos2word_matrix[next_pos_index]                                       #Retrieve the probabilities of words given the new POS tag
        
        
        #Loop through all previous words in the sentence
        for n, previous_words in enumerate(reversed(sentence[:-1])):                                #Sentence[:-1] omits the last word since that is already handled in previous line of code
            if n > 10:                                                                              #Max size for sentence is 10 words. Can be removed, sentence should not extend set length
                break                             
            
            #Multiply the probabilties of the potential words being n-step back the current word in the sentence 
            previous_word_index = self.vocabulary.word2indx(previous_words)
            transition_probs = self.word2word_matrices[n+1][previous_word_index]                             #NOTE: If there is a 0 frequency this might set the whole probability to 0 for that word. Need further investigation
            
            for i, value in enumerate(transition_probs):                                            #Work around for when a value in the matrix is 0
                if value > 0:
                    word_probabilities[i] *= value                                                  #Future note: Due to the sturcture of the matrices, this should multiply the values of the same words. Look into if this is actually correct
                                                                                                    #Future x2 note: This should be correct. The columns should be close to identical for the previous n-grams. Making i and value be the same word in this context

        
        #Bias words with the correct POS tag
        total_probs = word_probabilities * emission_probs


        #Prints the probabilities into text files 
        #self.save_debug_data(word_probabilities, emission_probs, total_probs, step_name=f"{sentence[-1]}")
        
        #Cases when there is no next word to transition to
        if np.sum(total_probs) == 0:
            next_pos_index = np.random.choice(len(next_pos_probs), p=next_pos_probs)
            self.current_pos_tag = self.indx_to_pos[next_pos_index]
            emission_probs = self.pos2word_matrix[next_pos_index]
            total_probs = word_probabilities * emission_probs

            if np.sum(total_probs) == 0:
                #print("No next word found")                                                      
                return None
        
        #Create probability distribution by normalizing counts 
        probabilities_sum = np.sum(total_probs)                                                    #Matrix should already have normalized values. However, seems to be edge cases where that is not the case thus this code is necessary
        if probabilities_sum != 1:                                                                 #Note: Finding a way to solve the problem and thus removing this code. Could yield better performance  
            total_probs = total_probs / probabilities_sum
        
        #next_word_index = np.random.choice(len(total_probs), p = total_probs)                      #Get next word, weighed on the calculated probabilites 
        next_word_index = np.argmax(total_probs)                                                    #Get next word with highest probability
        return self.vocabulary.indx2word(next_word_index)
    
    def print_sentence(self, sentence):
        sentence_str = ' '.join(sentence[0])
        print(sentence_str.capitalize(), sentence[1], "\n")
        return
    
    def save_debug_data(self, word_probabilities, emission_probs, total_probs, step_name):
        with open(os.path.join('Debug_matrix_prob', f"debug_data_{step_name}.txt"), "w") as file:
            file.write("Word Probabilities (non-zero entries):\n")
            np.savetxt(file, word_probabilities[word_probabilities > 0], fmt="%.6f", delimiter=",")

            file.write("\nEmission Probabilities (non-zero entries):\n")
            np.savetxt(file, emission_probs[emission_probs > 0], fmt="%.6f", delimiter=",")

            file.write("\nTotal Probabilities (non-zero entries):\n")
            np.savetxt(file, total_probs[total_probs > 0], fmt="%.6f", delimiter=",")

#------------------------------------------------MAIN------------------------------------------------

def main():
    if os.path.exists("saved_data.pkl"):
        pos_tags, pairfreq_tables, wordfreq, vocabulary, word_list = load_variables("saved_data.pkl")
        end_program = False
    else:
        start_animation()
        file_names = ["sample.txt", "A_room_with_a_view.txt", "half_first_quart.txt"]
        all_text_list = []
        '''
        for file_name in file_names:
            try:
                with open(file_name, 'r') as file:
                    while True:
                        chunk = file.read(4096)
                        if not chunk:
                            break
                        all_text_list.append(chunk)
            except FileNotFoundError:
                print(f"File '{file_name}' not found")
                return
            except IOError:
                print(f"Error reading the file '{file_name}'")
                return
        
        #all_text = ''.join(all_text_list)
        '''

        all_text = webtext.raw()
        
        tokenizer = TweetTokenizer(preserve_case = False)
        corpus = re.sub(r"[‘’´`]", "'", all_text)                                                                                              #Edge case where apostrophes can have different Unicode. This normalize them
        word_list = tokenizer.tokenize(corpus)                                                                                                  #Split text into words/tokens with help from nltk built in models. ( Will also tokenize symbols e.g ., [, & )      
        word_list = [token for token in word_list if re.match(r"^[A-Za-z]+(?:['-][A-Za-z]+)*$", token) and "_" not in token]                           #Clean tokens by removing special characters (edge case for "_" had to be included since it was not considered a character)
        pos_tags = pos_tag(word_list)                                                                                                           #Part-of-speech tagging. NLTK function that tags each word with a grammatical tag (Noun, verb, etc)

        vocabulary = CreateVocabulary(pos_tags)                                                                                                #pos_tags contain all information that word_list has with the added bonus of grammatical tags
        vocabulary.write_word2indx_to_file()
        
        wordfreq = create_frequency_table(word_list, vocabulary)                                                                               #Frequency of each word in the corpus 
        print_frequency_table(wordfreq, vocabulary)


        #List to store pairfrequency tables to aid in dynamic creation and handling of several tables
        pairfreq_tables = []                                            
        for n in range(2, 12):  
            pairfreq = PairFrequencyTable(word_list, n)
            pairfreq_tables.append(pairfreq)
            pairfreq.print_table(n)

        save_variables(pos_tags, pairfreq_tables, wordfreq, vocabulary, word_list)
        stop_animation()
        print(len(word_list))
        end_program = True

    #End program after running with new corpus. To make runtime more manageable 
    if end_program:
        return None 



    #Print the corpus size 
    #print(len(word_list))

    #Init the Markov Model
    num_sentences = 20
    generated_sentences = []
    scores = [] 
    text_generator = MarkovChain(pairfreq_tables, vocabulary, pos_tags, wordfreq)
   
    for _ in range(num_sentences):
        sentence, score = text_generator.generate_sentence()
        generated_sentences.append((sentence, score))

    #Sort the sentences by viterbi score 
    best_sentences = sorted(generated_sentences, key=lambda x: x[1], reverse=True)
    
    for i in range(0, 5):
        text_generator.print_sentence(best_sentences[i])

if __name__ == "__main__":
    download_punkt()
    prompt_for_corpus()
    main()