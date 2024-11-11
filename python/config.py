# path to all_words
all_words_path = 'data/all_words.txt'

# training parameters 
start_epoch = 56
start_batch = 0
batch_size = 25  # Related to how your batches are handled in the model (e.g., self.batch_size)

# data reader shuffle index list
load_list = False
index_list_file = 'data/shuffle_index_list'
cur_train_index = start_batch * batch_size
