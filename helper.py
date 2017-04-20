import numpy as np
import operator
from collections import defaultdict
import collections, sys
import logging
import re

class Vocab(object):
    unk = u'<unk>'
    bos = u'<bos>'
    eos = u'<eos>'
    def __init__(self, unk=True, bos=False, eos=False):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = self.unk
        if unk:
            self.add_word(self.unk, count=0)
        if bos:
            self.add_word(self.bos, count=0)
        if eos:
            self.add_word(self.eos, count=0)

    def add_word(self, word, count=1):
        word = word.strip()
        if len(word) == 0:
            return
        elif word.isspace():
            return
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))

    def limit_vocab_length(self, length):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            None
            
        Returns:
            None 
        """
        if length > self.__len__():
            return
        new_word_to_index = {self.unknown:0}
        new_index_to_word = {0:self.unknown}
        self.word_freq.pop(self.unknown)          #pop unk word
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        vocab_tup = sorted_tup[:length]
        self.word_freq = dict(vocab_tup)
        for word in self.word_freq:
            index = len(new_word_to_index)
            new_word_to_index[word] = index
            new_index_to_word[index] = word
        self.word_to_index = new_word_to_index
        self.index_to_word = new_index_to_word
        self.word_freq[self.unknown]=0

    def save_vocab(self, filePath):
        """
        Save vocabulary a offline file
        
        Args:
            filePath: where you want to save your vocabulary, every line in the 
            file represents a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        try:
            self.word_freq.pop(self.unknown)
            self.word_freq.pop(self.bos)
            self.word_freq.pop(self.eos)
        except:
            print 'warning, from save_vocab. no <unk>, <bos> or <eos>'
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        with open(filePath, 'wb') as fd:
            for (word, freq) in sorted_tup:
                fd.write(('%s\t%d\n'%(word, freq)).encode('utf-8'))

    def load_vocab_from_file(self, filePath, sep='\t'):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            filePath: vocabulary file path, every line in the file represents 
                a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        with open(filePath, 'rb') as fd:
            for line in fd:
                line_uni = line.decode('utf-8')
                word, freq = line_uni.split(sep)
                index = len(self.word_to_index)
                if word not in self.word_to_index:
                    self.word_to_index[word] = index
                    self.index_to_word[index] = word
                    self.word_freq[word] = int(freq)
            print 'load from <'+filePath+'>, there are {} words in dictionary'.format(len(self.word_freq))

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_to_index)


class SnliDataSet():

    def __init__(self, data_path, vocab, label_vocab):
        self.vocab=vocab
        self.label_vocab = label_vocab
        self.data = self.load_data(data_path)
        self.num_example = len(self.data)

    @staticmethod
    def load_data(fileName):
        import csv
        '''
        Args:
            fileName: from which file to load
            vocab: vocabulary dictionary
        Returns:
            data_list: list of sentence_list
        '''
        data_list = []
        with open(fileName) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for rcd in reader:
                label = rcd['gold_label'].decode('utf-8')
                snt1 = re.findall(r"[\w']+|[.,!?;]", rcd['sentence1'].decode('utf-8'))
                snt2 = re.findall(r"[\w']+|[.,!?;]", rcd['sentence2'].decode('utf-8'))
                if label == '-':
                    continue
                dataItem = (label, snt1, snt2)
                data_list.append(dataItem)
        return data_list

    def data_iter(self, batch_size, test=False):
        '''
        Args:
            data: list of sentence_list
            vocab: vocabulary
            batch_size: batch_size
        Returns:
            batch_x, batch_y, batch_len
        '''
        vocab = self.vocab
        label_vocab = self.label_vocab

        def batch_encodeNpad(dataList, vocab):
            sentLen = []
            data_matrix = []
            trunLen = max([len(o) for o in dataList])
            for wordList in dataList:
                length = len(wordList)
                if trunLen != 0:
                    length = min(length, trunLen)
                sentEnc = []
                if trunLen == 0:
                    for word in wordList:
                        sentEnc.append(vocab.encode(word))
                else:
                    for i in range(trunLen):
                        if i < length:
                            sentEnc.append(vocab.encode(wordList[i]))
                        else:
                            sentEnc.append(vocab.encode(vocab.unknown))
                sentLen.append(length)
                data_matrix.append(sentEnc)
            return sentLen, data_matrix

        def test_iter():
            data = self.data
            data_len = self.num_example
            epoch_size = data_len // batch_size
            idx = np.arange(data_len)
            for i in range(epoch_size+1):
                if i == epoch_size and epoch_size*batch_size<data_len:
                    indices = range(i * batch_size, data_len)
                else:
                    indices = range(i * batch_size, (i + 1) * batch_size)
                indices = idx[indices]

                label_ids = [label_vocab.encode(data[o][0]) for o in indices]
                snt1_matrix = [data[o][1] for o in indices]
                snt2_matrix = [data[o][2] for o in indices]
                snt1_len, snt1_matrix = batch_encodeNpad(snt1_matrix, vocab)
                snt2_len, snt2_matrix = batch_encodeNpad(snt2_matrix, vocab)
                yield label_ids, snt1_matrix, snt2_matrix, snt1_len, snt2_len

        def train_iter():
            data = self.data
            data_len = self.num_example
            epoch_size = data_len // batch_size
            idx = np.arange(data_len)
            np.random.shuffle(idx)
            for i in range(epoch_size):
                indices = range(i * batch_size, (i + 1) * batch_size)
                indices = idx[indices]

                label_ids = [label_vocab.encode(data[o][0]) for o in indices]
                snt1_matrix = [data[o][1] for o in indices]
                snt2_matrix = [data[o][2] for o in indices]
                snt1_len, snt1_matrix = batch_encodeNpad(snt1_matrix, vocab)
                snt2_len, snt2_matrix = batch_encodeNpad(snt2_matrix, vocab)
                yield label_ids, snt1_matrix, snt2_matrix, snt1_len, snt2_len

        if not test:
            return test_iter()
        else:
            return train_iter()

    @property
    def size(self):
        return self.num_example


def load_dataSet(vocab, label_vocab, train_path, valid_path, test_path):

    Datasets = collections.namedtuple('Datasets', ['train', 'valid', 'test'])
    train_data = SnliDataSet(train_path, vocab, label_vocab)
    test_data = SnliDataSet(test_path, vocab, label_vocab)
    valid_data = SnliDataSet(valid_path, vocab, label_vocab)
    return Datasets(train=train_data, valid=valid_data, test=test_data)


def ids2text(ids, vocab):
    '''
    Args:
        ids: numpy matrix shape(b_sz, tstp) type of int32
    '''
    eos_id = vocab.word_to_index[vocab.eos]
    sentences = []
    for sent_id in ids:
        sent = ' '.join([vocab.decode(o) for o in sent_id if o != eos_id])
        sentences.append(sent)
    return sentences


def readEmbedding(fileName):
    """
    Read Embedding Function
    
    Args:
        fileName : file which stores the embedding
    Returns:
        embeddings_index : a dictionary contains the mapping from word to vector
    """
    embeddings_index = {}
    with open(fileName, 'r') as f:
        for line in f:
            line_uni = line.strip()
            line_uni = line_uni.decode('utf-8')
            values = line_uni.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

'''Read and make embedding'''

def mkEmbedMatrix(embed_dic, vocab_dic):
    """
    Construct embedding matrix
    
    Args:
        embed_dic : word-embedding dictionary
        vocab_dic : word-index dictionary
    Returns:
        embedding_matrix: return embedding matrix
    """
    if type(embed_dic) is not dict or type(vocab_dic) is not dict:
        raise TypeError('Inputs are not dictionary')
    if len(embed_dic) < 1 or len(vocab_dic) <1:
        raise ValueError('Input dimension less than 1')
    
    EMBEDDING_DIM = len(embed_dic.items()[0][1])
    #embedding_matrix = np.zeros((len(vocab_dic), EMBEDDING_DIM), dtype=np.float32)
    embedding_matrix = np.random.rand(len(vocab_dic), EMBEDDING_DIM).astype(np.float32) * 0.05
    valid_mask = np.ones(len(vocab_dic), dtype=np.bool)
    for word, i in vocab_dic.items():
        embedding_vector = embed_dic.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            valid_mask[i] = False
    return embedding_matrix, valid_mask


'''accuracy and confusion calculating'''

def pred_from_prob_single(prob_matrix):
    """

    Args:
        prob_matrix: probability matrix have the shape of (data_num, class_num),
            type of float. Generated from softmax activation

    Returns:
        ret: return class ids, shape of(data_num,)
    """
    ret = np.argmax(prob_matrix, axis=1)
    return ret


def calculate_accuracy_single(pred_ids, label_ids):
    """
    Args:
        pred_ids: prediction id list shape of (data_num, ), type of int
        label_ids: true label id list, same shape and type as pred_ids

    Returns:
        accuracy: accuracy of the prediction, type float
    """
    if np.ndim(pred_ids) != 1 or np.ndim(label_ids) != 1:
        raise TypeError('require rank 1, 1. get {}, {}'.format(np.rank(pred_ids), np.rank(label_ids)))
    if len(pred_ids) != len(label_ids):
        raise TypeError('first argument and second argument have different length')

    accuracy = np.mean(np.equal(pred_ids, label_ids))
    return accuracy


def calculate_confusion_single(pred_list, label_list, label_size):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((label_size, label_size), dtype=np.int32)
    for i in xrange(len(label_list)):
        confusion[label_list[i], pred_list[i]] += 1

    tp_fp = np.sum(confusion, axis=0)
    tp_fn = np.sum(confusion, axis=1)
    tp = np.array([confusion[i, i] for i in range(len(confusion))])

    precision = tp.astype(np.float32)/(tp_fp+1e-40)
    recall = tp.astype(np.float32)/(tp_fn+1e-40)
    overall_prec = np.float(np.sum(tp))/(np.sum(tp_fp)+1e-40)
    overall_recall = np.float(np.sum(tp))/(np.sum(tp_fn)+1e-40)

    return precision, recall, overall_prec, overall_recall, confusion


def print_confusion_single(prec, recall, overall_prec, overall_recall, num_to_tag):
    """Helper method that prints confusion matrix."""
    logstr="\n"
    logstr += '{:15}\t{:7}\t{:7}\n'.format('TAG', 'Prec', 'Recall')
    for i, tag in sorted(num_to_tag.items()):
        logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format(tag.encode('utf-8'), prec[i], recall[i])
    logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format('OVERALL', overall_prec, overall_recall)
    logging.info(logstr)
    print logstr


def ConstructVocab(data_path):
    vocab = Vocab()
    label_vocab = Vocab()

    train_data = SnliDataSet.load_data(data_path + '/snli_1.0_train.txt')
    dev_data = SnliDataSet.load_data(data_path + '/snli_1.0_dev.txt')
    test_data = SnliDataSet.load_data(data_path + '/snli_1.0_test.txt')

    def construct_block(data_list):
        for dataItem in data_list:
            label_vocab.add_word(dataItem[0])
            for word in dataItem[1]:
                vocab.add_word(word)
            for word in dataItem[2]:
                vocab.add_word(word)
    for data_list in (train_data, dev_data, test_data):
        construct_block(data_list)

    vocab.save_vocab(data_path+'/vocab.all')
    vocab.limit_vocab_length(100000)
    vocab.save_vocab(data_path+'/vocab.100k')

    label_vocab.save_vocab(data_path+'/id2tag.txt')



if __name__ == '__main__':
    ConstructVocab('/home/jjgong/.jjgong/myData/snli_1.0/raw')

