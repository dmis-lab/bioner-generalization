import argparse
import os
import re
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from collections import Counter
from tqdm import tqdm
from string import punctuation
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 


class TextPreprocess():
    """
    Text Preprocess module that supportt lowercase, removing punctuation, typo correction
    This code is from the BioSyn repository (Sung et al., ACL 2020)
    """
    def __init__(self, 
            lowercase=True, 
            remove_punctuation=True,
            ignore_punctuations="",
            stemming=False,
            typo_path=None):
        """
        Parameters
        ==========
        typo_path : str
            path of known typo dictionary
        """
        self.lowercase = lowercase
        self.typo_path = typo_path
        self.rmv_puncts = remove_punctuation
        self.punctuation = punctuation
        for ig_punc in ignore_punctuations:
            self.punctuation = self.punctuation.replace(ig_punc,"")
        self.rmv_puncts_regex = re.compile(r'[\s{}]+'.format(re.escape(self.punctuation)))
        
        self.stemming = stemming
        if self.stemming:
            self.stemmer = PorterStemmer() 

        if typo_path:
            self.typo2correction = self.load_typo2correction(typo_path)
        else:
            self.typo2correction = {}

    def load_typo2correction(self, typo_path):
        typo2correction = {}
        with open(typo_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                s = line.strip()
                tokens = s.split("||")
                value = "" if len(tokens) == 1 else tokens[1]
                typo2correction[tokens[0]] = value    

        return typo2correction 

    def remove_punctuation(self,phrase):
        phrase = self.rmv_puncts_regex.split(phrase)
        phrase = ' '.join(phrase).strip()

        return phrase

    def correct_spelling(self, phrase):
        phrase_tokens = phrase.split()
        phrase = ""

        for phrase_token in phrase_tokens:
            if phrase_token in self.typo2correction.keys():
                phrase_token = self.typo2correction[phrase_token]
            phrase += phrase_token + " "
       
        phrase = phrase.strip()
        return phrase
    
    
    def stem_tokens(self, text):
        words = word_tokenize(text) 
        
        out = []
        for w in words:
            out.append(self.stemmer.stem(w))
        out = " ".join(out)
        return out

    def run(self, text):
        if self.lowercase:
            text = text.lower()

        if self.typo_path:
            text = self.correct_spelling(text)

        if self.rmv_puncts:
            text = self.remove_punctuation(text)
        
        if self.stemming:
            text = self.stem_tokens(text)

        text = text.strip()

        return text

    def add_space(self, s):
        if not s: return ""
        result = s[0]
        prev_c = s[0]
        for c in s[1:]:
            if prev_c in punctuation:
                result += " "
                result += c
            else:
                if c in punctuation:
                    result += " "
                    result += c
                else:
                    result += c
            prev_c = c
        result = " ".join(result.split())
        return result

def get_single_cuis(cuis):
    if type(cuis) == str: cuis = [cuis]
    elif type(cuis) == list: cuis = cuis

    cui_list = []
    for c_i, cc in enumerate(cuis):
        if '|' in cc:
            cs = cc.split('|')
            for c in cs:
                cui_list.append(c)
        elif '+' in cc:  # For NCBI
            cs = cc.split('+')
            for c in cs:
                cui_list.append(c)
        else:
            cui_list.append(cc)
    return cui_list


def update(spl, preprocessor, entity_str, index_tmp, num_mem, num_syn, num_con, mention_dictionary, cui_dictionary, test_splits, cuis):
    if spl == 'Mem':
        if (preprocessor.run(entity_str) in mention_dictionary):
            num_mem += 1
        else:
            for j in index_tmp:
                test_splits[spl][j] = 'O'
    elif spl == 'Syn':
        if (preprocessor.run(entity_str) not in mention_dictionary) and \
            sum([1 if c in cui_dictionary else 0 for c in get_single_cuis(cuis[index_tmp[0]])]) > 0:
            num_syn += 1
        else:
            for j in index_tmp:
                test_splits[spl][j] = 'O'
    elif spl == 'Con':
        if ((preprocessor.run(entity_str) not in mention_dictionary) and \
            sum([1 if c in cui_dictionary else 0 for c in get_single_cuis(cuis[index_tmp[0]])]) == 0):
            num_con += 1
        else:
            for j in index_tmp:
                test_splits[spl][j] = 'O'
    else:
        raise ValueError("Invalid name: {}.".format(spl))

    return num_mem, num_syn, num_con, test_splits

def partition(preprocessor, mention_dictionary, cui_dictionary, test_splits, tokens, cuis):
    num_mem = num_syn = num_con = 0
    for spl in list(test_splits.keys()):
        if spl == 'Overall': 
            continue
            
        # init
        entity_tmp = []
        index_tmp = []
        inside_entity = False
        i = -1
        for pred in tqdm(test_splits[spl]):
            i += 1

            if pred[0] == 'B':
                if inside_entity:
                    assert cuis[index_tmp[0]] != '-'
                    entity_str = ' '.join(entity_tmp)
                    
                    num_mem, num_syn, num_con, test_splits = update(spl, preprocessor, entity_str, index_tmp, num_mem, num_syn, num_con, \
                                                            mention_dictionary, cui_dictionary, test_splits, cuis)
                    # init
                    inside_entity = False
                    entity_tmp = []
                    index_tmp = []
                    
                    inside_entity = True
                    entity_tmp.append(tokens[i])
                    index_tmp.append(i)
                else:
                    inside_entity = True
                    entity_tmp.append(tokens[i])
                    index_tmp.append(i)
            elif pred[0] == 'I':
                inside_entity = True
                entity_tmp.append(tokens[i])
                index_tmp.append(i)
            elif pred[0] == 'O':
                if inside_entity:
                    assert cuis[index_tmp[0]] != '-'
                    entity_str = ' '.join(entity_tmp)
                    
                    num_mem, num_syn, num_con, test_splits = update(spl, preprocessor, entity_str, index_tmp, num_mem, num_syn, num_con, \
                                                            mention_dictionary, cui_dictionary, test_splits, cuis)
                    # init
                    inside_entity = False
                    entity_tmp = []
                    index_tmp = []
    print("Splits | Mem: {}, Syn: {}, Zero: {}".format(num_mem, num_syn, num_con))
    return test_splits

def print_score(SPLITS, test_splits, predictions):
    print("\n--Evaluation--")

    for spl in SPLITS:
        if spl == 'Overall':
            p = precision_score([test_splits['Overall']], [predictions])
            r = recall_score([test_splits['Overall']], [predictions])
            f1 = f1_score([test_splits['Overall']], [predictions])
            print("{} {:2.1f}\t{:2.1f}\t{:2.1f}".format(spl, p*100, r*100, f1*100))
        else:
            r = recall_score([test_splits[spl]], [predictions])
            print("{} {:2.1f}".format(spl, r*100))

def main():
    # arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mention_dictionary', type=str, required=True)
    parser.add_argument('--cui_dictionary', type=str, required=True)
    parser.add_argument('--gold_labels', type=str, required=True)
    parser.add_argument('--gold_cuis', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    
    args = parser.parse_args()

    preprocessor = TextPreprocess()
    
    # Load dictionaries
    with open(args.mention_dictionary) as g:
        mention_dictionary = [preprocessor.run(line.rstrip('\n')) for line in g.readlines()]
    with open(args.cui_dictionary) as g:
        cui_dictionary = [line.rstrip('\n') for line in g.readlines()]

    print(len(mention_dictionary), len(set(mention_dictionary)))

    # Load model predictions
    f = open(args.predictions)
    preds = f.readlines()
    
    # Load test data
    g = open(args.gold_labels)
    gold_labels = g.readlines()
    g = open(args.gold_cuis)
    gold_cuis = g.readlines()
    
    # Initialize
    SPLITS = ['Overall', 'Mem', 'Syn', 'Con']
    test_splits = {}
    for l in SPLITS:
        test_splits[l] = []

    #
    predictions = []
    tokens = []
    cuis = []
    for i, (pred, label) in enumerate(zip(preds, gold_labels)):
        if not pred.split(): continue
        
        p_token = pred.split()[0]
        p_label = pred.split()[1]
        l_token = label.split()[0]
        c_token = gold_cuis[i].split()[0]
        assert p_token == l_token
        assert c_token == l_token
        
        l_label = label.split()[1]
        
        # The seqeval framework requires entity types for entity-level NER evaluation.
        # Assign 'MISC' to all annotations if the data is of a single type and the annotations do not specify an entity type.
        if p_label == 'B' or p_label == 'I':
            p_label = p_label + '-MISC'
        if l_label == 'B' or l_label == 'I':
            l_label = l_label + '-MISC'
        
        cuis.append(gold_cuis[i].split()[1])
        predictions.append(p_label)
        tokens.append(p_token)
        for spl in SPLITS:
            test_splits[spl].append(l_label)
    
    # Partition benchmarks
    test_splits = partition(preprocessor, mention_dictionary, cui_dictionary, test_splits, tokens, cuis)
    
    # Evaluation
    print_score(SPLITS, test_splits, predictions)

if __name__ == '__main__':
    main()

