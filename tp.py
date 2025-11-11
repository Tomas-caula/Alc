from ALC import *
import numpy as np
import os 


def cargarCarpeta(carpeta):
    embeddingsGatos = np.load( os.path.join(carpeta, 'train', 'cats', 'efficientnet_b3_embeddings.npy'))
    embeddingsPerros = np.load( os.path.join(carpeta, 'train', 'dogs', 'efficientnet_b3_embeddings.npy'))

    embeddingGatosVal = np.load( os.path.join(carpeta, 'val', 'cats', 'efficientnet_b3_embeddings.npy'))
    embeddingPerrosVal = np.load( os.path.join(carpeta, 'val', 'dogs', 'efficientnet_b3_embeddings.npy'))

    Xv = np.concatenate((embeddingGatosVal, embeddingPerrosVal))
    Yv = np.concatenate((
        np.zeros((embeddingGatosVal.shape[0], 1)),
        np.ones((embeddingPerrosVal.shape[0], 1))
    ))

    Xt = np.concatenate((embeddingsGatos, embeddingsPerros))

    Yt = np.concatenate((
        np.zeros((embeddingsGatos.shape[0], 1)),
        np.ones((embeddingsPerros.shape[0], 1))
    ))
    return Xt, Yt, Xv, Yv
    
Xt, Yt = cargarCarpeta('./cats_and_dogs/')
print(Xt)
print(Yt)