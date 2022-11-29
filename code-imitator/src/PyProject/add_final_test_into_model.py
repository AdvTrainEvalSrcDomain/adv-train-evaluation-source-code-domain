import pickle
import sys
from featureextractionV2.StyloFeaturesProxy import StyloFeaturesProxy
from featureextractionV2.StyloFeatures import StyloFeatures
from featureextractionV2.StyloUnigramFeatures import StyloUnigramFeatures

model = pickle.load(open(sys.argv[1], 'rb'))  #model path
trainfiles: StyloFeatures = model.data_final_train
#print(model.data_final_test)
unigrammmatrix_train = trainfiles.codestyloreference
stop_words_codestylo = ["txt", "in", "out", "attempt0", "attempt", "attempt1", "small", "output", "input"]
unigrammmatrix_test = StyloUnigramFeatures(inputdata=sys.argv[2],  #testing set
                                                    nocodesperprogrammer=8,
                                                    noprogrammers=204,   #number of authors in testset
                                                    binary=False, tf=True, idf=True,
                                                    ngram_range=(1, 3), stop_words=stop_words_codestylo,
                                                    trainobject=unigrammmatrix_train)
testfiles: StyloFeatures = StyloFeaturesProxy(codestyloreference=unigrammmatrix_test)
testfiles.createtfidffeatures(trainobject=trainfiles)
testfiles.selectcolumns(index=None, trainobject=trainfiles)
model.data_final_test = testfiles

with open(sys.argv[1], 'wb') as f:
    pickle.dump(model, f)
