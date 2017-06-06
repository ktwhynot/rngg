import random
import numpy as np

def getAlphabet(fromme):
    """
    Takes a string and returns a sorted array of the unique characters present in the string.
    Note that upper/lower-case letters are considered different (eg. A and a).
    Eg. Given "Ryan is a dick", returns ['R', 'a', 'c', 'd', 'i', 'k', 'n', 's', 'y', ' ']
    """
    alphabet = []
    for c in fromme:
        newchar = True
        for a in alphabet:
            if a == c:
                newchar = False
                break
        if newchar:
            alphabet.append(c)

    alphabet.sort()
    return alphabet

def getEncodedStr(toencode, alphabet):
    """
    Take a string and a list of characters, and encode each character in the string
    as the character's position in the list. Eg. Given "cab" and ['a','b','c'], you would get
    [2,0,1].

    Expects all characters in the string to be present in the list.
    """
    temp = []
    for c in toencode:
        temp.append(alphabet.index(c))
    return temp

def oneHVector(theindex, n):
    """
    Take an index and a number n, and returns a numpy array with n-1 zeroes and a single 1 at the given index
    eg. Given 3 and 5, you would get [0 0 0 1 0] (remember that indices start at 0)

    Expects the index < n.
    """
    temp = np.zeros(n)
    temp[theindex] = 1
    return temp

def oneHToIndex(thevector):
    """
    A wrapper to take a numpy array and return the index with the highest value.
    All this does is call argmax. Having a different function name makes the code
    a bit more understandable, though may add needless overhead.
    """
    return np.argmax(thevector)

def trainDataMaker(alphabet, encodedstr):
    """
    Create the training data for the LSTM given the alphabet and index-encoded raw data.
    One instance of training data is a one-hot-encoded character as input, and the next OH-encoded character in the sequence as output.
    eg. In an alphabet of [a b c] and a string of "cab", the input for a single instance would be the OH-encoding of c: [0 0 1]
    and the expected output the OH-encoding of a: [1 0 0].

    This function does the above for the entire string; forming the numpy arrays of OH-encoded characters for both inputs and outputs.
    These are then reshaped to meet Keras' specifications, and returned as a tuple: (inputs, outputs).

    Expects each index represented by the ecnoded string to be present in the given alphabet
    """
    x_train = np.array(np.mat(np.zeros(len(alphabet))))
    y_train = np.array(np.mat(oneHVector(encodedstr[0], len(alphabet))))
    for c in range(0, len(encodedstr)-1):
        x_train = np.vstack([x_train, oneHVector(encodedstr[c], len(alphabet))])
        y_train = np.vstack([y_train, oneHVector(encodedstr[c+1], len(alphabet))])

    x_train = x_train.reshape(len(x_train),1,len(alphabet))

    return (x_train, y_train)

def shuffleNames(thenames):
    """
    Take a single string of whitespace-separated names and return a string of those same names in a random order.
    The returned string will have each name separated by a newline character.
    """
    namelist = thenames.split()
    random.shuffle(namelist)
    #Since split removed the whitespace, we need to add it back in after each name
    returnme = ""
    for name in namelist:
        returnme += name
        returnme += "\n"

    return returnme
