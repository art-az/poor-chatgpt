import pickle

def save_variables(pos_tags, pairfreq_tables, wordfreq, vocabulary, word_list, filepath="saved_data.pkl"):
    with open(filepath, "wb") as f:
        pickle.dump({"pos_tags": pos_tags, "pairfreq_tables": pairfreq_tables, "wordfreq": wordfreq, "vocabulary": vocabulary, "word_list": word_list }, f)

def load_variables(filepath="saved_data.pkl"):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["pos_tags"], data["pairfreq_tables"], data["wordfreq"], data["vocabulary"], data["word_list"]