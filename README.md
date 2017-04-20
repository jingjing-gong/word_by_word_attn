## Implementation of word-by-word attention in "Reasoning about Entailment with Neural Attention" Paper.

## Requirement:

Python 2.7

Tensorflow 1.0.0 

Recommend OS: Linux


## How to run:

Fill `all_data` directory with snli data, and preferably fill `all_data/embed` directory with glove pretrained embedding, change the embedding file name to `embedding.N` (`N` is the embedding dimension, for example, `embedding.100` means this is a embedding file that store the 100dim embedding)

When all the data is in place, you can train the model as follows:
        
	# generate a config file in save01 dir, if there already is a config, 
	# the following generation is not necessary
	# config file is used to configure the structure of the model, can be edited
	# and load existing config file without generating one
	
	python model.py --weight-path ./savings/save01 
	
	# load config file in ./savings/save01, and start training
	
	python model.py --weight-path ./savings/save01 --load-config
	
	# load the config file in ./savings/save01 along with saved parameters and start run test
	
	python model.py --weight-path ./savings/save01 --load-config --train-test test

