from ALC import *
import numpy as np


def cargarCarpeta(carpeta):
    embeddingsGatos = np.load( carpeta + 'train/cats/efficientnet_b3_embeddings.npy')
    embeddingsPerros = np.load( carpeta + 'train/dogs/efficientnet_b3_embeddings.npy')

    embeddingGatosVal = np.load( carpeta + 'val/cats/efficientnet_b3_embeddings.npy')
    embeddingPerrosVal = np.load(carpeta + 'val/dogs/efficientnet_b3_embeddings.npy')

    Xv = np.concatenate((embeddingGatosVal, embeddingPerrosVal))
    Yv = np.concatenate((
        np.array([np.array([1, 0])] * embeddingGatosVal.shape[0]), #Gatos es (1,0)
        np.array([np.array([0, 1])] * embeddingPerrosVal.shape[0])
    ))

    Xt = np.concatenate((embeddingsGatos, embeddingsPerros))

    Yt = np.concatenate((
        np.array([np.array([1, 0])] * embeddingsGatos.shape[0]),
        np.array([np.array([0, 1])] * embeddingsPerros.shape[0])
    ))
    return Xt, Yt, Xv, Yv
    
Xt, Yt, Xv, Yv = cargarCarpeta('./cats_and_dogs/')


def pinvEcuacionesNormales(X, L, Y):
    