import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class GMP:

    def __init__(self, user_num, item_num):

        latent_features = 8

        # User embedding
        user = Input(shape=(1,), dtype='int32')
        user_embedding = Embedding(user_num, latent_features, input_length=user.shape[1])(user)
        user_embedding = Flatten()(user_embedding)

        # Item embedding
        item = Input(shape=(1,), dtype='int32')
        item_embedding = Embedding(item_num, latent_features, input_length=item.shape[1])(item)
        item_embedding = Flatten()(item_embedding)

        # Merge
        concatenated = Multiply()([user_embedding, item_embedding])

        # Output
        output_layer = Dense(1, kernel_initializer='lecun_uniform', name='output_layer')(concatenated) # 1,1 / h(8,1)초기화

        # Model
        self.model = Model([user, item], output_layer)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def get_model(self):
        model = self.model
        return model





