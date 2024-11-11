from __future__ import print_function
import cPickle as pickle
import time
import re
import numpy as np
from gensim.models import word2vec, KeyedVectors

WORD_VECTOR_SIZE = 300

raw_movie_conversations = open('data/movie_conversations.txt', 'r').read().split('\n')[:-1]

utterance_dict = pickle.load(open('data/utterance_dict', 'rb'))

ts = time.time()
corpus = word2vec.Text8Corpus("data/tokenized_all_words.txt")
word_vector = word2vec.Word2Vec(corpus, size=WORD_VECTOR_SIZE)
word_vector.wv.save_word2vec_format(u"model/word_vector.bin", binary=True)
word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
print("Time Elapsed: {} secs\n".format(time.time() - ts))

""" Extract only the vocabulary part of the data """
def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words)
    return data

ts = time.time()
conversations = []
print('len conversation', len(raw_movie_conversations))
con_count = 0
traindata_count = 0
for conversation in raw_movie_conversations:
    conversation = conversation.split(' +++$+++ ')[-1]
    conversation = conversation.replace('[', '')
    conversation = conversation.replace(']', '')
    conversation = conversation.replace('\'', '')
    conversation = conversation.split(', ')
    assert len(conversation) > 1
    for i in range(len(conversation)-1):
        con_a = utterance_dict[conversation[i+1]].strip()
        con_b = utterance_dict[conversation[i]].strip()
        if len(con_a.split()) <= 22 and len(con_b.split()) <= 22:
            con_a = [refine(w) for w in con_a.lower().split()]
            # con_a = [word_vector[w] if w in word_vector else np.zeros(WORD_VECTOR_SIZE) for w in con_a]
            conversations.append((con_a, con_b))
            traindata_count += 1
    con_count += 1
    if con_count % 1000 == 0:
        print('con_count {}, traindata_count {}'.format(con_count, traindata_count))
pickle.dump(conversations, open('data/reversed_conversations_lenmax22', 'wb'), True)
print("Time Elapsed: {} secs\n".format(time.time() - ts))