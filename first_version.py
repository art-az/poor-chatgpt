import os
from save_file_utils import save_variables, load_variables
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import pos_tag
from nltk.util import ngrams
from nltk.corpus import webtext
from nltk.corpus import gutenberg
from nltk.corpus import reuters
from datasets import load_dataset
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
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
        nltk.data.find("corpora/gutenberg")
        nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download("webtext")
        nltk.download("gutenberg")
        nltk.download("reuters")

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
        """
        #Insert custom tag SPC for special token
        if "EOS" in self.word2indx_list:
            index, _ = self.word2indx_list["EOS"]
            self.word2indx_list["EOS"] = (index, ["SPC_eos"])

        if "BOS" in self.word2indx_list:
            index, _ = self.word2indx_list["BOS"]
            self.word2indx_list["BOS"] = (index, ["SPC_bos"])
        """

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
        self.table = defaultdict(defaultdict(int).copy)                                    #Key is the first word. Value is another dict where the key is the word pair and value it's frequency. [Word]-[WordPair]:[Frequency]
        #self.n = n_step                                                                    #Variable to identify what kind of table it is. ex: n = 3, Frequency table of 3 steps
        self.create_pair_frequency_table(word_list, n_step)

    def create_pair_frequency_table(self, word_list, n_step, exclude_self_pairs=True):                               #Initilizes frequency table by generating ngrams and storing the first and last word in each tuple as a pair
        ngram_list = ngrams(word_list, n_step)
        for ngram_tuple in ngram_list:
            first_word, last_word = ngram_tuple[0], ngram_tuple[-1]                         #[-1] will access last element in list
            if exclude_self_pairs and n_step == 2 and first_word == last_word:              #in 2-grams skip if word is paired with itself
                continue
            self.table[first_word][last_word] += 1

    def print_table(self, iteration = 2):
        directory = "pair_frequencies"
        filename = f"pair_frequencies_{iteration}.txt"
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as file:
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
            



#-------------------------------Main implementation for text generation-------------------------------# 

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


###---------------Initialization of Markov Model---------------###


    def build_wordpair_matrix(self):
        vocab_size = len(self.vocabulary.word2indx_list)                                            #Amount of unique words in the vocabulary
        transition_matrices = []

        #Initilize the matrixes and store them in a list 
        for _ in range(15):
            #transition_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)                                  #Create a matrix filled with zeros (2D NumPy array) 
            #transition_matrices.append(transition_matrix)
            transition_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float64)
            transition_matrices.append(transition_matrix)

        #Loop through each word and it's index, in the vocabulary 
        for word, (index, _) in self.vocabulary.word2indx_list.items():                          
            
            for n in range(15):
                following_words = self.pairfreq_tables[n].table.get(word, None)                     #Get all pair-words for the current word
            
                if following_words:
                    total = sum(following_words.values())                                           #Calculate the total frequency of all pair-words/following words to the current word
                    for next_word, count in following_words.items():                                #Loop through each pair-word and it's frequency 
                        next_index = self.vocabulary.word2indx(next_word)                           #Get the index of the current pair-word (For storage/look-up optimization)
                        transition_matrices[n][index, next_index] = count / total                   #Store the current word (Row) and current pair-word (Column), and the probability to transition to the pair-word

        csr_matrices = [matrix.tocsr() for matrix in transition_matrices]
        
        #Size debugger
        data_size = csr_matrices[1].data.nbytes
        indices_size = csr_matrices[1].indices.nbytes
        indptr_size = csr_matrices[1].indptr.nbytes

        total_size_bytes = data_size + indices_size + indptr_size
        total_size_bytes = total_size_bytes/1000000
        print("Data size (bytes):", data_size)
        print("Indices size (bytes):", indices_size)
        print("Indptr size (bytes):", indptr_size)
        print("Total sparse matrix size (MB):", total_size_bytes)

        return csr_matrices

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

        #Count frequency for each POS:word pair
        for word, tag in pos_tags:
            word_indx = self.vocabulary.word2indx(word)
            pos_indx = self.pos_to_indx[tag]
            emission_matrix[pos_indx][word_indx] += 1

        #Apply Laplace Smoothing
        alpha = 0.00001                                                                              #Laplace smoothing parameter 
        row_sums = emission_matrix.sum(axis=1, keepdims=True)
        emission_matrix = (emission_matrix + alpha) / (row_sums + alpha * vocab_size)

        #Normalize rows after previous adjustments 
        #emission_matrix = emission_matrix/row_sums

        return emission_matrix



###--------------------Start of Text Generation--------------------###



    def viterbi_init(self, start_word=None):

        if not start_word:
            start_word = self.generate_starting_word_pos(self.wordfreq)     #CALLS OLD FUNCTION

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
                        viterbi_POS_path[i] = current_tag
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

#--------------------EOS handling--------------------#

    #Linearly interpolate probability of EOS token based on sentence length (longer sentence = higher probability)
    def eos_multiplier(self, sentence_length, min_length = 5, max_length = 9, max_multiplier = 4):

        #Repress EOS if sentence is shorter than 5 words
        if sentence_length < min_length:
            return 0
        
        #Boost EOS if sentence is 9 words or more
        elif sentence_length >= max_length:
            return max_multiplier
        
        else:
            return max_multiplier * (sentence_length - min_length) / (max_length - min_length)
        
    def adjust_eos_probability(self, probabilities, sentence_length):
        
        eos_index = self.vocabulary.word2indx("EOS")
        multiplier = self.eos_multiplier(sentence_length)

        adjusted_prob = probabilities
        adjusted_prob[eos_index] *= multiplier
        
        return adjusted_prob

#--------------------Starting Word Calculations--------------------# 

    #Hard-coded probabilities for the starting POS-tag in a sentence (Based on general observations of sentence structures, can be adjusted)
    def starting_pos_tag(self):
        start_pos_tags = ['DT', 'PRP', 'NN', 'NNP', 'JJ', 'CC', 'IN', 'RB', 'VB']
        start_probabilities = [0.25, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0.025, 0.025]

        start_tag = np.random.choice(start_pos_tags, p=start_probabilities)

        return start_tag
    
    #Find start word by only including words that match with starting POS tag and pick at random, with higher probability based on frequency 
    def generate_starting_word_pos(self, wordfreq):
        self.current_pos_tag = self.starting_pos_tag()
        
        words_with_tag = []
        for word in wordfreq.keys():
            if self.current_pos_tag in self.vocabulary.word_pos_tag(self.vocabulary.indx2word(word)) and self.vocabulary.indx2word(word) != "EOS":
                words_with_tag.append(word)

        if not words_with_tag:                                                              #if no words with POS tag found. Fall back on frequency based pick
            words_with_tag = list(wordfreq.keys())
            print("Words with POS tag not found. Fall back on frequency based pick for start word")

        freq_weights = [wordfreq[word] for word in words_with_tag] 
        initial_word = random.choices(words_with_tag, weights=freq_weights)[0]

        return initial_word

    def calculate_first_word_bos(self):
        
        #Retrieve BOS index
        bos_index = self.vocabulary.word2indx("BOS")
        if bos_index is None:
            raise ValueError("BOS token not found in vocabulary")
        
        #Words following BOS tokens
        word_probs = self.word2word_matrices[0][bos_index].toarray().flatten()

        #Set EOS probability to 0 since it is not a word 
        eos_index = self.vocabulary.word2indx("EOS")
        if eos_index is not None:
            word_probs[eos_index] = 0

        #Retrieve specific POS-tag for BOS token
        if "SPC_bos" not in self.pos_to_indx:
            raise ValueError("BOS POS-tag not found in dictonary lookup")
        bos_pos_index = self.pos_to_indx["SPC_bos"]

        #Zero array of word2word-matrix size 
        combined_scores = np.zeros_like(word_probs)
        vocab_size = len(word_probs)
        best_pos_tags = [None] * vocab_size                                                 #To store the best tag calculated for each candidate tag
        
        #For each candidate word in word_probs. Compute a score 
        for i in range(vocab_size):
            p_word = word_probs[i]
            if p_word == 0:
                continue                                                                     #Skip 0 probability words

            word = self.vocabulary.indx2word(i)
            candidate_tags = self.vocabulary.word_pos_tag(word)
            best_tag_score = 0
            chosen_tag = None

            for tag in candidate_tags:
                tag_index = self.pos_to_indx[tag]

                #P(tag|BOS) -POS2POS
                p_tag_given_bos = self.pos2pos_matrix[bos_pos_index, tag_index]

                #P(word|tag) -POS2Word
                p_word_given_bos = self.pos2word_matrix[tag_index, i]

                #Score P(tag|BOS) * P(word|tag) and store best score 
                tag_score = p_tag_given_bos * p_word_given_bos

                if tag_score > best_tag_score:
                    best_tag_score = tag_score
                    chosen_tag = tag

            #Store the chosen tag for this word
            best_pos_tags[i] = chosen_tag

            #Combine best tag score with BOS-to-word transition probability
            combined_scores[i] = p_word * best_tag_score


        #Normalize probabilities 
        total = np.sum(combined_scores)
        if total > 0:
            combined_scores /= total
        else:
            raise ValueError("Zero probability when normalizing combined_scores for BOS token")


        next_word_index = np.random.choice(len(combined_scores), p = combined_scores)
        first_word = self.vocabulary.indx2word(next_word_index)
        self.current_pos_tag = best_pos_tags[next_word_index]

        return first_word

#--------------------Text Generation--------------------#
  
    def generate_sentence_beam(self, beam_width=5, max_length=15):

        first_word = self.calculate_first_word_bos()
        first_pos = self.current_pos_tag

        initial_log_score = np.log(1e-12 + 1)                                                 #Dummy value of 1, for now

        #Init beam with first word
        beam = [([first_word], [first_pos], initial_log_score)]

        #Storage for completed sentences (ending with EOS)
        completed = []


        #Beam expansion
        for _ in range(max_length - 1):
            new_beam = []

            for seq, pos_seq, log_score in beam:
                
                #If sequence ends with EOS, add to completed and skip current expansion
                if seq[-1] == "EOS":
                    completed.append((seq[:-1], pos_seq[:-1], log_score))
                    continue
                
                #Retrive probability distribution for next words based on current word
                next_word_probs = self.get_next_word(seq, False)
                if next_word_probs is None:
                    raise ValueError("get_next_word returned None")

                #Iterate all candidate words from the prob distribution
                for candidate_word_index in range(len(next_word_probs)):

                    can_word_prob = next_word_probs[candidate_word_index]                                                   #Probability of current word
                    if can_word_prob == 0:
                        continue

                    candidate_word = self.vocabulary.indx2word(candidate_word_index)
                    candidate_tags = self.vocabulary.word_pos_tag(candidate_word)
                    best_tag_score = 0
                    chosen_tag = None

                    #Use the last pos tag in the sentence
                    current_pos = pos_seq[-1]
                    if current_pos not in self.pos_to_indx:
                        continue                                                             #Failsafe is POS does not exist in database

                    current_pos_index = self.pos_to_indx[current_pos]

                    #Evaluate each POS for current candidate word
                    for tag in candidate_tags:
                        if tag not in self.pos_to_indx:
                            continue

                        tag_index = self.pos_to_indx[tag]

                        #Probability this tag follows current POS in sentence
                        p_tag = self.pos2pos_matrix[current_pos_index, tag_index]

                        #Probability of current word given current tag
                        p_word_with_tag = self.pos2word_matrix[tag_index, candidate_word_index]

                        #Calculate tag score and store highest score
                        tag_score = p_tag * p_word_with_tag
                        if tag_score > best_tag_score:
                            best_tag_score = tag_score
                            chosen_tag = tag

                    #Combine candidate word and calculated best tag
                    combined_prob = can_word_prob * best_tag_score
                    if combined_prob <= 0:
                        continue

                    #Update score with the new combined probability
                    new_log_score = log_score + np.log(combined_prob)

                    #Create new candidate sentence
                    new_seq = seq + [candidate_word]
                    new_pos_seq = pos_seq + [chosen_tag]

                    #Insert into new beam
                    new_beam.append((new_seq, new_pos_seq, new_log_score))

            #If no new candidates were generated. Break the loop
            if not new_beam:
                break
            
            #Sort candidate seq by log score in descending order. Filtering out all but the top 5 
            beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:beam_width]
        
        final_candidates = beam + completed
        if not final_candidates:
            raise ValueError("No sentences generated (generate_Sentence_beam)")

        best_candidate = max(final_candidates, key=lambda x :x[2])
        final_sentence, final_pos_seq, final_log_score = best_candidate 

        return final_sentence, final_log_score

    #The start and main function for generating text
    def generate_sentence(self, start_word=None, length=15):
        
        if not start_word:                                                                    #Default start word when no input
            start_word = self.calculate_first_word_bos()

        current_word = start_word
        sentence = [current_word]
        
        pos_tag_sequence = []                                                                 #Debug: print purpose 
        pos_tag_sequence.append(self.current_pos_tag)
        
        #Generate sentence word for word 
        for _ in range(length-1):
            
            current_word = self.get_next_word(sentence)
            pos_tag_sequence.append(self.current_pos_tag)

            if current_word == "EOS":
                break

            elif current_word:              
                sentence.append(current_word)
           
        
        #pos_str = ' '.join(pos_tag_sequence)
        #print(pos_str)                                                                        #Debug: print the POS tags for the sentence
        
        #Returns score of how matching the POS sequence is to the computed POS sequence of the viterbi algorithm
        score = self.viterbi_scoring(sentence, pos_tag_sequence)
        
        return sentence, score                                                                       #Return sentence (currently unused)

    #Called in generate_sentence_beam()
    def get_next_word(self, sentence, return_word = True):
        
        last_word_index = self.vocabulary.word2indx(sentence[-1])
        if last_word_index is None:
            raise ValueError(f"Token '{sentence[-1]}' not found in vocabulary.")
        word_probabilities = self.word2word_matrices[0][last_word_index].toarray().flatten()                           #Returns the row of probabilites for word pairs corresponding to the last word in the sentence 

        #Get next POS tag
        current_pos_index = self.pos_to_indx[self.current_pos_tag]
        next_pos_probs = self.pos2pos_matrix[current_pos_index]
        next_pos_index = np.random.choice(len(next_pos_probs), p=next_pos_probs)                    #Pick next pos tag weighed by the probabilities in pos_transition_matrix (Alt: pick POS with highest prob)
        #next_pos_index = np.argmax(next_pos_probs)
        
        self.current_pos_tag = self.indx_to_pos[next_pos_index]                                     #Save the new POS tag        
        emission_probs = self.pos2word_matrix[next_pos_index]                                       #Retrieve the probabilities of words given the new POS tag
        
        
        #Loop through all previous words in the sentence
        for n, previous_words in enumerate(reversed(sentence[:-1])):                                #Sentence[:-1] omits the last word since that is already handled in previous line of code
            if n + 1 >= len(self.word2word_matrices):                                                                              #Max size for sentence is 10 words. Can be removed, sentence should not extend set length
                break                             
            
            #Multiply the probabilties of the potential words being n-step back the current word in the sentence 
            previous_word_index = self.vocabulary.word2indx(previous_words)
            transition_probs = self.word2word_matrices[n+1][previous_word_index].toarray().flatten()                             #NOTE: If there is a 0 frequency this might set the whole probability to 0 for that word. Need further investigation
            
            for i, value in enumerate(transition_probs):                                            #Work around for when a value in the matrix is 0
                if value > 0:
                    word_probabilities[i] *= value                                                  #Future note: Due to the sturcture of the matrices, this should multiply the values of the same words. Look into if this is actually correct
                                                                                                    #Future x2 note: This should be correct. The columns should be close to identical for the previous n-grams. Making i and value be the same word in this context

        
        #Bias words with the "correct" POS tag
        total_probs = word_probabilities * emission_probs

        #Inflates/deflates the EOS token depending on sentence length 
        total_probs = self.adjust_eos_probability(total_probs, len(sentence))
        
        #Edge Case: when there is no next word to transition to
        if np.sum(total_probs) == 0:
            next_pos_index = np.random.choice(len(next_pos_probs), p=next_pos_probs)
            self.current_pos_tag = self.indx_to_pos[next_pos_index]
            emission_probs = self.pos2word_matrix[next_pos_index]
            total_probs = word_probabilities * emission_probs
            print("Resorted to fallback word")
            if np.sum(total_probs) == 0:
                print("No next word found")                                                      
                return None
        
        #Create probability distribution by normalizing counts 
        probabilities_sum = np.sum(total_probs)                                                    #Matrix should already have normalized values. However, seems to be edge cases where that is not the case thus this code is necessary
        if probabilities_sum != 1:                                                                 #Note: Finding a way to solve the problem and thus removing this code. Could yield better performance  
            total_probs /= probabilities_sum
        

        """Prints the probabilities into text files"""
        #self.save_debug_data(word_probabilities, emission_probs, total_probs, step_name=f"{sentence[-1]}")
        

        #Return the probability distribution for all candidate next words instead of sampling the next word
        if not return_word:
            return total_probs

        #next_word_index = np.random.choice(len(total_probs), p = total_probs)                        #Get next word, weighed on the calculated probabilites 
        next_word_index = np.argmax(total_probs)                                                      #Get next word with highest probability
        return self.vocabulary.indx2word(next_word_index)
    


#--------------------Misc Functions--------------------#



    def print_sentence(self, sentence):
        sentence_str = ' '.join(sentence[0])
        print(sentence_str.capitalize(), f"({sentence[1]:.2f})", "\n")
        return
    
    def save_debug_data(self, word_probabilities, emission_probs, total_probs, step_name):
        with open(os.path.join('Debug_matrix_prob', f"debug_data_{step_name}.txt"), "w") as file:
            file.write("Word Probabilities (non-zero entries):\n")
            np.savetxt(file, word_probabilities[word_probabilities > 0], fmt="%.16f", delimiter=",")

            file.write("\nEmission Probabilities (non-zero entries):\n")
            np.savetxt(file, emission_probs[emission_probs > 0], fmt="%.16f", delimiter=",")

            file.write("\nTotal Probabilities (non-zero entries):\n")
            np.savetxt(file, total_probs[total_probs > 0], fmt="%.16f", delimiter=",")

    def save_pos2pos_matrix(self, filename="pos2pos_transition_matrix.txt"):
        np.savetxt(filename, self.pos2pos_matrix, fmt="%.8f", delimiter=", ")

    def save_pos2word_matrix(self, filename="pos2word_emission_matrix.txt"):
        np.savetxt(filename, self.pos2word_matrix, fmt="%.16f", delimiter=", ")




###------------------------------------------------MAIN------------------------------------------------###

def main():
    if os.path.exists("saved_data.pkl"):
        start_animation()
        pos_tags, pairfreq_tables, wordfreq, vocabulary, word_list, text_generator = load_variables("saved_data.pkl")
        stop_animation()

    else:
        start_animation()

        '''
        file_names = ["sample.txt", "A_room_with_a_view.txt", "half_first_quart.txt"]
        all_text_list = []
        
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
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        #all_text = "\n".join(dataset["text"])
        #all_text = ''.join([str(webtext.raw()), str(gutenberg.raw())])
        
        dataset_text = "\n".join(["\n".join(dataset["text"]), webtext.raw(), gutenberg.raw()])
        sentences = nltk.sent_tokenize(dataset_text)
        sentences = [f"<BOS> {single_sentence} <EOS>" for single_sentence in sentences]                                                               #Tokenize text to sentences and add EOS token at end of each sentence
        all_text = "\n".join(sentences)

        tokenizer = TreebankWordTokenizer()
        corpus = re.sub(r"[‘’´`]", "'", all_text)                                                                                               #Edge case where apostrophes can have different Unicode. This normalize them
        word_list = tokenizer.tokenize(corpus)                                                                                                  #Split text into words/tokens with help from nltk built in models. ( Will also tokenize symbols e.g ., [, & )      
        #word_list = [token for token in word_list if re.match(r"^[A-Za-z]+(?:['-][A-Za-z]+)*$", token) and "_" not in token]                           #Clean tokens by removing special characters (edge case for "_" had to be included since it was not considered a character)
        word_list = [token for token in word_list if re.match(r"^(?:(?:<EOS>|<BOS>)|[A-Za-z]+(?:['-][A-Za-z]+)*)$", token) and "_" not in token]
        pos_tags = pos_tag(word_list)                                                                                                           #Part-of-speech tagging. NLTK function that tags each word with a grammatical tag (Noun, verb, etc)
        
        #Input special tags for special tokens
        pos_tags = [(word, "SPC_eos") if word in ("<EOS>", "EOS") 
                    else (word, "SPC_bos") if word in ("<BOS>", "BOS")
                    else (word, tag)    for word, tag in pos_tags]

        vocabulary = CreateVocabulary(pos_tags)                                                                                                #pos_tags contain all information that word_list has with the added bonus of grammatical tags
        vocabulary.write_word2indx_to_file()
        
        wordfreq = create_frequency_table(word_list, vocabulary)                                                                               #Frequency of each word in the corpus 
        print_frequency_table(wordfreq, vocabulary)


        #List to store pairfrequency tables to aid in dynamic creation and handling of several tables
        pairfreq_tables = []                                            
        for n in range(2, 17):  
            pairfreq = PairFrequencyTable(word_list, n)
            pairfreq_tables.append(pairfreq)
            pairfreq.print_table(n)

        text_generator = MarkovChain(pairfreq_tables, vocabulary, pos_tags, wordfreq)
        text_generator.save_pos2pos_matrix()
        text_generator.save_pos2word_matrix()

        save_variables(pos_tags, pairfreq_tables, wordfreq, vocabulary, word_list, text_generator)
        stop_animation()
        print(len(word_list))
    
        return
    #End program after running with new corpus. To make runtime more manageable 
   

    
    print("Generating text")
    
    #Generate text with the MarkovChain object 
    num_sentences = 10
    generated_sentences = []
    scores = [] 
    
    
    for _ in range(num_sentences):
        sentence, score = text_generator.generate_sentence_beam()
        generated_sentences.append((sentence, score))
    
    #Sort the sentences by viterbi score 
    best_sentences = sorted(generated_sentences, key=lambda x: x[1], reverse=True)
    
    for i in range(0, 10):
        text_generator.print_sentence(best_sentences[i])
    


if __name__ == "__main__":
    download_punkt()
    prompt_for_corpus()
    main()