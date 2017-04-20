import ConfigParser
import json
class Config(object):
    """Holds model hyperparams and data information.
    
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    """General"""
    train_data='./all_data/snli_1.0_train.txt'
    val_data='./all_data/snli_1.0_dev.txt'
    test_data='./all_data/snli_1.0_test.txt'
    vocab_path='./all_data/vocab.all'
    id2tag_path='./all_data/id2tag.txt'
    embed_path='./all_data/embed/embedding.'
    
    cnn_after_embed = True
    
    neural_model = "dummy"
    pre_trained = False
    batch_size = 256
    embed_size = 200
    max_epochs = 50
    early_stopping = 5
    dropout = 0.9
    lr = 0.01
    decay_steps = 500
    decay_rate = 0.9
    class_num = 0
    reg = 0.2
    num_steps = 40
    fnn_layers = 2
    
    """lstm"""
    hidden_size = 300
    rnn_numLayers=1
    
    """cnn"""
    num_filters = 128
    filter_sizes = [3, 4, 5]
    cnn_numLayers=1
    
    def saveConfig(self, filePath):

        cfg = ConfigParser.ConfigParser()
        cfg.add_section('General')
        cfg.add_section('lstm')
        cfg.add_section('cnn')
        
        cfg.set('General', 'train_data', self.train_data)
        cfg.set('General', 'val_data', self.val_data)
        cfg.set('General', 'test_data', self.test_data)
        cfg.set('General', 'vocab_path', self.vocab_path)
        cfg.set('General', 'id2tag_path', self.id2tag_path)
        cfg.set('General', 'embed_path', self.embed_path)
        
        cfg.set('General', 'cnn_after_embed', self.cnn_after_embed)

        cfg.set('General', 'neural_model', self.neural_model)
        cfg.set('General', 'pre_trained', self.pre_trained)
        cfg.set('General', 'batch_size', self.batch_size)
        cfg.set('General', 'embed_size', self.embed_size)
        cfg.set('General', 'max_epochs', self.max_epochs)
        cfg.set('General', 'early_stopping', self.early_stopping)
        cfg.set('General', 'dropout', self.dropout)
        cfg.set('General', 'lr', self.lr)
        cfg.set('General', 'decay_steps', self.decay_steps)
        cfg.set('General', 'decay_rate',self.decay_rate)
        cfg.set('General', 'class_num', self.class_num)
        cfg.set('General', 'reg', self.reg)
        cfg.set('General', 'num_steps', self.num_steps)
        cfg.set('General', 'fnn_layers', self.fnn_layers)
        
        cfg.set('lstm', 'hidden_size', self.hidden_size)
        cfg.set('lstm', 'rnn_numLayers', self.rnn_numLayers)
        
        cfg.set('cnn', 'num_filters', self.num_filters)
        cfg.set('cnn', 'filter_sizes', self.filter_sizes)
        cfg.set('cnn', 'cnn_numLayers', self.cnn_numLayers)
        
        with open(filePath, 'w') as fd:
            cfg.write(fd)
        
    def loadConfig(self, filePath):

        cfg = ConfigParser.ConfigParser()
        cfg.read(filePath)
        
        self.train_data = cfg.get('General', 'train_data')
        self.val_data = cfg.get('General', 'val_data')
        self.test_data = cfg.get('General', 'test_data')
        self.vocab_path = cfg.get('General', 'vocab_path')
        self.id2tag_path = cfg.get('General', 'id2tag_path')
        self.embed_path = cfg.get('General', 'embed_path')
        
        self.cnn_after_embed = cfg.getboolean('General', 'cnn_after_embed')
        
        self.neural_model = cfg.get('General', 'neural_model')
        self.pre_trained = cfg.getboolean('General', 'pre_trained')
        self.batch_size = cfg.getint('General', 'batch_size')
        self.embed_size = cfg.getint('General', 'embed_size')
        self.max_epochs = cfg.getint('General', 'max_epochs')
        self.early_stopping = cfg.getint('General', 'early_stopping')
        self.dropout = cfg.getfloat('General', 'dropout')
        self.lr = cfg.getfloat('General', 'lr')
        self.decay_steps = cfg.getint('General', 'decay_steps')
        self.decay_rate = cfg.getfloat('General', 'decay_rate')
        self.class_num = cfg.getint('General', 'class_num')
        self.reg = cfg.getfloat('General', 'reg')
        self.num_steps = cfg.getint('General', 'num_steps')
        self.fnn_layers = cfg.getint('General', 'fnn_layers')

        self.hidden_size = cfg.getint('lstm', 'hidden_size')
        self.rnn_numLayers = cfg.getint('lstm', 'rnn_numLayers')
        
        self.num_filters = cfg.getint('cnn', 'num_filters')
        self.filter_sizes = json.loads(cfg.get('cnn', 'filter_sizes'))
        self.num_filters = cfg.getint('cnn', 'cnn_numLayers')

