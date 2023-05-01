import nltk
from fastai.text.all import *
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from collections import defaultdict
from NE_info import NE_list

import torch
torch.cuda.set_device(2)

from argparse import ArgumentParser

# tokenizer from https://stackoverflow.com/questions/58105967/spacy-tokenization-of-hyphenated-words
def custom_tokenizer(nlp):
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

# sourced from https://github.com/JHL-HUST/PWWS/blob/dfd9f6b9a8fb9a57e3a0b9207e1dcbe8a624618c/paraphrase.py
def get_pos(word):
    '''Wordnet POS tag'''
    pos = word.tag_[0].lower()
    if pos in ['r', 'n', 'v']:  # adv, noun, verb
        return pos
    elif pos == 'j':
        return 'a'  # adj
    
def evaluate_word_saliency(learn, x, sentence_list, pos_neg):  
    word_saliency_array = []
    # Compute word saliency (difference in classification accuracy if word replace with <unk>)
    for i, word in enumerate(sentence_list):
        copy_list = sentence_list.copy()
        copy_list[i] = "<UNK>"
        new_sentence = " ".join(copy_list)
        x_i = learn.predict(new_sentence)[2][pos_neg]
        word_saliency = x - x_i
        word_saliency_array.append((word, word_saliency))
    return word_saliency_array

def find_best_synonym(learn, sentence_list, index, word, x, pos_neg):
    """Should return synonym to replace it and the change in classification probability that it causes"""
    
    # https://www.askpython.com/python/examples/pos-tagging-in-nlp-using-spacy
    synonyms = defaultdict(list)

    token = word.text
    part_of_speech = get_pos(word)

    # create a list of synonyms of the word based on part of speech
    for syn in wordnet.synsets(token, pos=part_of_speech):
        for i in syn.lemmas():
            synonyms[token].append(i.name())

    # go through all the synonyms and assess how much they change classification
    max_synonym = ""
    max_prob_difference = 0
    for s in synonyms[token]:
        if '_' in s:
            s.replace('_', ' ')
        copy_list = sentence_list.copy()
        copy_list[index] = s
        new_sentence = " ".join(copy_list)
        x_hat_i = learn.predict(new_sentence)[2][pos_neg]
        prob_difference = (x - x_hat_i).item()
        if (prob_difference > max_prob_difference):
            max_prob_difference = prob_difference
            max_synonym = s

    return (max_synonym, max_prob_difference)    

# defined in Eq.(8) from PWWS paper
def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def calculate_score(word_saliency, prob_difference):
    return word_saliency * prob_difference

def main():
    
    argparser = ArgumentParser("Get adversarial examples")
    argparser.add_argument("model_name", type=str, help="name of saved model")
    argparser.add_argument("model_path", type=str, help="path of saved model")
    argparser.add_argument("data_path", type=str, help="path of data")
    args = argparser.parse_args()
 

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = custom_tokenizer(nlp) # need custom tokenizer cus spacy nlp pipe doesn't properly tokenize hyphenated words
    
    # load in model
    learn = load_learner(args.model_path, cpu=False)
    
    # load in data
    df = pd.read_csv(args.data_path) # Note: model will autotokenize text
    text = []
    correct_category = []        
        
    # find adversaries
    adverse_text = []
    total_subs = []
    for t in df.itertuples():
        sentence = t[1]
        sentence_list = [t.text for t in nlp(sentence)] # to ensure proper tokenization
        pos_neg = t[2]

        # the base probility of the correct classification of the sentence
        initial_prediction = learn.predict(sentence)
        x = initial_prediction[2][pos_neg] # second index depends if pos or neg example 0 -> prob of neg, 1 -> prob of pos

        # find word saliency of all words
        word_saliency_array = evaluate_word_saliency(learn, x, sentence_list, pos_neg)
        word_saliency_vector = np.array([pair[1] for pair in word_saliency_array])
        word_saliency_vector = softmax(word_saliency_vector)

        # spacy piping
        sentence = nlp(sentence)

        word_scores =  []

        for index, word in enumerate(sentence):

            # check for NEs
            NE_candidates = NE_list[pos_neg]
            NE_tags = list(NE_candidates.keys())
            NER_tag = word.ent_type_
            
            word_saliency = word_saliency_vector[index]
            
            if NER_tag in NE_tags:
                synonym = NE_candidates[NER_tag]
                copy_list = sentence_list.copy()
                copy_list[index] = synonym
                new_sentence = " ".join(copy_list)
                x_hat_i = learn.predict(new_sentence)[2][pos_neg]
                prob_difference = (x - x_hat_i).item()
            else:
                # Get set of synonyms from wordnet
                # return value: (synonym to replace the word with, probability change caused by replacement)
                synonym_pair = find_best_synonym(learn, sentence_list, index, word, x, pos_neg)
                synonym = synonym_pair[0]
                prob_difference = synonym_pair[1]

            if (prob_difference > 0): # maybe get rid of this if-statement
                # use word saliency and change in probability to calculate the score 
                score = calculate_score(word_saliency, prob_difference)
                word_scores.append((index, synonym, score))

        # order word replacements by score
        word_scores.sort(key = itemgetter(2), reverse=True)

        # replace words in sentence until the prediction is false
        new_list = sentence_list.copy()
        adversarial_sentence = ""
        subs = 0
        for (word_index, synonym, _) in word_scores:
            new_list[word_index] = synonym
            subs += 1
            new_sentence = " ".join(new_list)
            new_prediction = learn.predict(new_sentence)
            if (initial_prediction[0] != new_prediction[0]):
                adversarial_sentence = new_sentence
                adverse_text.append(adversarial_sentence)
                text.append(t[1])
                correct_category.append(pos_neg)
                total_subs.append(subs)
                print("success")
                break
    
    # save results
    file = open("./results.txt", "a")
    num_examples = len(adverse_text)
    accuracy = num_examples / 10
    file.write(f"\n{args.model_name}: {num_examples} adversarial examples made of 10 sentences\n {accuracy}%")
    print(accuracy)
    file.close()
    
    d = {"text":text, "adverse_text":adverse_text, "total_subs": total_subs, "category": correct_category}
    df_adverse = pd.DataFrame(data=d)
    pathfile = f"../data/{args.model_name}_adverse_examples10.csv"
    df_adverse.to_csv(pathfile, index=False)
    
    
if __name__ == "__main__":
    main()