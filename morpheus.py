from abc import ABCMeta, abstractmethod
import lemminflect, random
import torch
from nltk.tag.perceptron import PerceptronTagger
from nltk.tag.mapping import map_tag
from rouge_score import rouge_scorer
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from sacremoses import MosesTokenizer, MosesDetokenizer

'''
MorpheusBase contains methods common to all subclasses.
Users should call `Morpheus<Task><Model>().morph()` to generate adversarial examples. 
All concrete classes must have the `morph` method implemented, either by itself or its parent.
'''
class MorpheusSeq2Seq(metaclass=ABCMeta):
    def __init__(self):
        self.tagger = PerceptronTagger()

    def morph(self, source, reference, constrain_pos=True):
        orig_tokenized = MosesTokenizer(lang='en').tokenize(source)
        pos_tagged = [(token, map_tag("en-ptb", 'universal', tag))
                      for (token, tag) in self.tagger.tag(orig_tokenized)]
        pos_tagged = [(tagged[0], '.') if '&' in tagged[0] else tagged for tagged in pos_tagged]
        token_inflections = self.get_inflections(orig_tokenized, pos_tagged, constrain_pos)
        original_score, orig_predicted = self.get_score(source, reference) #model pred 1
        print('original_score ', original_score)

        forward_perturbed, forward_score, \
        forward_predicted, num_queries_forward = self.search_seq2seq(token_inflections, 
                                                                     orig_tokenized,
                                                                     source,
                                                                     original_score,
                                                                     reference)

        if forward_score == original_score:
            forward_predicted = orig_predicted

        if forward_score == 0:
            return MosesDetokenizer(lang='en').detokenize(forward_perturbed), forward_predicted, num_queries_forward + 1

        backward_perturbed, backward_score, \
        backward_predicted, num_queries_backward = self.search_seq2seq(token_inflections, 
                                                                       orig_tokenized,
                                                                       source,
                                                                       original_score,
                                                                       reference,
                                                                       backward=True)

        if backward_score == original_score:
            backward_predicted = orig_predicted
        num_queries = 1 + num_queries_forward + num_queries_backward
        if forward_score < backward_score:
            return MosesDetokenizer(lang='en').detokenize(forward_perturbed), forward_predicted, num_queries
        else:
            return MosesDetokenizer(lang='en').detokenize(backward_perturbed), backward_predicted, num_queries

    def search_seq2seq(self, token_inflections, orig_tokenized, original,
                       original_score, reference, backward=False):
        perturbed_tokenized = orig_tokenized.copy()

        max_score = original_score
        num_queries = 0
        max_predicted = ''

        if backward:
            token_inflections = reversed(token_inflections)
        
        detokenizer = MosesDetokenizer(lang='en')

        for curr_token in token_inflections:
            max_infl = orig_tokenized[curr_token[0]]
            for infl in curr_token[1]:
                perturbed_tokenized[curr_token[0]] = infl
                perturbed = detokenizer.detokenize(perturbed_tokenized)
                curr_score, predicted = self.get_score(perturbed, reference) #model pred 2 pred - spanish, pert - english, ref - spanish
                num_queries += 1
                if curr_score < max_score:
                    max_score = curr_score
                    max_infl = infl
                    max_predicted = predicted
            perturbed_tokenized[curr_token[0]] = max_infl
        return perturbed_tokenized, max_score, max_predicted, num_queries
    
    @abstractmethod
    def get_score(self, source, reference):
        pass

    @staticmethod
    def get_inflections(orig_tokenized, pos_tagged, constrain_pos):
        have_inflections = {'NOUN', 'VERB', 'ADJ'}
        token_inflections = [] 

        for i, word in enumerate(orig_tokenized):
            lemmas = lemminflect.getAllLemmas(word)

            if lemmas and pos_tagged[i][1] in have_inflections:
                if pos_tagged[i][1] in lemmas:
                    lemma = lemmas[pos_tagged[i][1]][0]
                else:
                    lemma = random.choice(list(lemmas.values()))[0]

                if constrain_pos:
                    inflections = (i, list(set([infl for tup in lemminflect.getAllInflections(lemma, upos=pos_tagged[i][1]).values() for infl in tup])))
                else:
                    inflections = (i, list(set([infl for tup in lemminflect.getAllInflections(lemma).values() for infl in tup])))
                random.shuffle(inflections[1])
                token_inflections.append(inflections)
                
        return token_inflections

class MorpheusHuggingfaceSeq2seq(MorpheusSeq2Seq):
    def __init__(self, model_path, rouge_type='rougeL', max_input_tokens=1024, use_cuda=True):
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        self.rouge_type = rouge_type
        if torch.cuda.is_available() and use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path,config=config)
        self.model.eval()
        self.model.to(self.device)
        self.max_input_tokens = max_input_tokens
        super().__init__()

    def get_score(self, source, reference):
        print('source ', source)
        predicted = self.model_predict(source)
        print('predicted ', predicted)
        return self.scorer.score(predicted, reference)[self.rouge_type].fmeasure, predicted

    def model_predict(self, source):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenized = self.tokenizer.encode(source, max_length=self.max_input_tokens, return_tensors='pt')
        generated = self.model.generate(tokenized.to(device))
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    