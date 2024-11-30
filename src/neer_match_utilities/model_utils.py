from pathlib import Path
import dill
# from pymlsem import scoder_model

def load_model(directory:Path, model_name:str):
    pass
#     # similarity map
#     with open(directory / model_name / 'similarity_map.pkl', 'rb') as fp:
#         similarity_map = dill.load(fp)

#     # initialize model
#     model = scoder_model.MatchingModel(similarity_map)

#     # load weights
#     model.load_weights(directory / model_name / 'model_{model_name}'.format(model_name=model_name))

#     return model, similarity_map