#SETTINGS
#The relative path for the list of names to train on
file_to_read_from = "Lovecraftian Names.txt"
#The amount of names to generate
amount_of_names = 10
#Whether or not to clean the generated names
#This will remove things such as sub-3-letter names, as well as ensure each name is capitalized and unique
#May lead to an infinite loop if you overfit the training data or cannot generate enough unique names
clean_output = True
#The relative filepath to save the generated names to
file_to_save_names = "Generated Names v2.txt"
#The relative filepath to load a precompiled (and likely pretrained) model from
#Set to an empty string to not load any model
#Note that if epoch_sets is not set to 0, this model will be trained
#Should be an h5 file
load_model_path = ""
#The amount of epochs to train before shuffling the names
#Set to 0 to not shuffle (may lead to overfitting)
epochs_to_shuffle = 3
#The amount of times to repeat the above shuffling process
#The total amount of epochs in training will be epochs_to_shuffle x epoch_sets
#When not shuffling, this will be the total epochs
#Set to 0 to not train (eg. if you're using a pre-trained model)
epoch_sets = 50
#Path to save the model after training
#Should be an h5 file
#Set to an empty string to not save the model
save_model_path = "Lovecraftian Name Generator v2.h5"


import Preprocessor as ppr
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Dense, Activation, LSTM


thedata = ""

with open (file_to_read_from, "r") as thefile:
    thedata = thefile.read()

#Determine the unique characters in a string
alphabet = ppr.getAlphabet(thedata)

#Encode each character in a string as an index in the alphabet
encodedstr = ppr.getEncodedStr(thedata, alphabet)

#Convert the encoded string to training data
x_train,y_train = ppr.trainDataMaker(alphabet, encodedstr)

#Create the LSTM
model = Sequential()
if load_model_path != "":
    model = load_model(load_model_path)
    print("Loaded model from", load_model_path)
else:
    model.add(LSTM(256, stateful=True, return_sequences=True, batch_input_shape=(1,1,len(alphabet))))
    model.add(LSTM(256))
    model.add(Dense(len(alphabet), activation='softmax'))
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

#Train it
if epoch_sets > 0:
    if epochs_to_shuffle == 0:
        model.fit(x_train, y_train, epochs=epoch_sets, batch_size=1, shuffle=False, verbose=2)
    else:
        for i in range(0, epoch_sets):
            print("Epoch set:",i+1,"/",epoch_sets)
            model.fit(x_train, y_train, epochs=epochs_to_shuffle, batch_size=1, shuffle=False, verbose=2)
            #Shuffle the names
            newnames = ppr.shuffleNames(thedata)
            newestr = ppr.getEncodedStr(newnames, alphabet)
            x_train,y_train = ppr.trainDataMaker(alphabet, newestr)

if save_model_path != "":
    model.save(save_model_path)
    print("Model saved to", save_model_path)

#Generate some names
generated_names = []
namecounter = 0
originalnames = thedata.split()
while namecounter < amount_of_names:
    thename = ""
    accept_name = True
    thechar = np.array(np.mat(np.zeros(len(alphabet))))
    thechar = thechar.reshape(len(thechar),1,len(alphabet))
    newline = False
    while not newline:
        temp = model.predict(thechar, batch_size=1, verbose=0)
        newchar = alphabet[ppr.oneHToIndex(temp)]
        if newchar == '\n':
            newline = True
        else:
            thename += newchar
        thechar = temp.reshape(len(temp),1,len(alphabet))
    if clean_output:
        if len(thename) < 3:
            thename = ""
            accept_name = False
        else:
            if thename[0].islower():
                thename = thename[0].upper() + thename[1:]
            #Check if there's a duplicate in either the original names or new names
            if thename in originalnames or thename in generated_names:
                accept_name = False
            
    if accept_name:
        generated_names.append(thename)
        namecounter += 1

#Save the names
with open(file_to_save_names, "a+") as thefile:
    for name in generated_names:
        print(name, file=thefile)
    print("Names saved in",file_to_save_names)


