from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# intereply encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

# define NMT model
def define_model(vocab, timesteps, n_units):
	model = Sequential()
	model.add(Embedding(vocab, n_units, input_length=timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(vocab, activation='softmax')))
	return model

# load datasets
dataset = load_clean_sentences('both.pkl')
dataset1 = dataset.reshape(-1,1)
train = load_clean_sentences('train.pkl')
test = load_clean_sentences('test.pkl')

# prepare tokenizer
all_tokenizer = create_tokenizer(dataset1[:,0])
all_vocab_size = len(all_tokenizer.word_index) + 1
all_length = max_length(dataset1[:, 0])
print('ALL Vocabulary Size: %d' % (all_vocab_size))
print('ALL Max question length: %d' % (all_length))

# prepare training data
trainX = encode_sequences(all_tokenizer, all_length, train[:, 0])
trainY = encode_sequences(all_tokenizer, all_length, train[:, 1])
trainY = encode_output(trainY, all_vocab_size)

# prepare validation data
testX = encode_sequences(all_tokenizer, all_length, test[:, 0])
testY = encode_sequences(all_tokenizer, all_length, test[:, 1])
testY = encode_output(testY, all_vocab_size)

# define model
model = define_model(all_vocab_size, all_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# summarize defined model
print(model.summary())

#train and save model
model.fit(trainX, trainY, epochs=500, batch_size=64, validation_data=(testX, testY), verbose=1)
filename = 'model1.h5'
model.save(filename)