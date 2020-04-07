import numpy, sys
import tensorflow as tf


def function():
    # load the network weights
    filename = "Models/weights-improvement-35-1.2661.h5"
    model = tf.keras.models.load_model(filename)

    # load ascii text and covert to lowercase
    filename = "DiscussionCorrect.txt"
    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()
    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    n_chars = len(raw_text)
    n_vocab = len(chars)
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)

    # pick a random seed
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    # generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")