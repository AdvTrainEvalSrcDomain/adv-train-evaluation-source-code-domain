import pickle
import sys
import numpy as np
from featureextractionV2.StyloFeaturesProxy import StyloFeaturesProxy
from featureextractionV2.StyloFeatures import StyloFeatures
from featureextractionV2.StyloUnigramFeatures import StyloUnigramFeatures
import sklearn.feature_selection


orig_model = pickle.load(open(sys.argv[1], 'rb'))  #pre-trained model
model = pickle.load(open(sys.argv[2], 'rb'))  #fine-tuned model
trainfiles: StyloFeatures = orig_model.data_final_train

unigrammmatrix_train = trainfiles.codestyloreference

model.data_final_train.codestyloreference.cvectorizer = trainfiles.codestyloreference.cvectorizer
model.data_final_train.codestyloreference._tfidftransformer = trainfiles.codestyloreference._tfidftransformer
model.data_final_train.codestyloreference._selected_features_indices = trainfiles.codestyloreference._selected_features_indices
#setattr(model, 'data_full_train', trainfiles)

with open(sys.argv[2], 'wb') as f:
    pickle.dump(model, f)
