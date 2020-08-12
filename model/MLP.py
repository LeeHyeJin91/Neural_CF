import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class MLP:

    def __init__(self, user_num, item_num):

        # User embedding
        user = Input(shape=(1,), dtype='int32')
        user_embedding = Embedding(user_num, 32, input_length=user.shape[1])(user)
        user_embedding = Flatten()(user_embedding)

        # Item embedding
        item = Input(shape=(1,), dtype='int32')
        item_embedding = Embedding(item_num, 32, input_length=item.shape[1])(item)
        item_embedding = Flatten()(item_embedding)

        # Merge
        concatenated = Concatenate()([user_embedding, item_embedding])
        dropout = Dropout(rate=0.2)(concatenated)

        # Layer1
        layer_1 = Dense(units=64, activation='relu', name='layer1')(dropout)  # (64,1)
        dropout1 = Dropout(rate=0.2, name='dropout1')(layer_1)                # (64,1)
        batch_norm1 = BatchNormalization(name='batch_norm1')(dropout1)        # (64,1)

        # Layer2
        layer_2 = Dense(units=32, activation='relu', name='layer2')(batch_norm1)  # (32,1)
        dropout2 = Dropout(rate=0.2, name='dropout2')(layer_2)                    # (32,1)
        batch_norm2 = BatchNormalization(name='batch_norm2')(dropout2)            # (32,1)

        # Layer3
        layer_3 = Dense(units=16, activation='relu', name='layer3')(batch_norm2)  # (16,1)

        # Layer4
        layer_4 = Dense(units=8, activation='relu', name='layer4')(layer_3)  # (8,1)

        # Output
        output_layer = Dense(1, kernel_initializer='lecun_uniform', name='output_layer')(layer_4)  # (1,1) / h(8,1)초기화

        # Model
        self.model = Model([user, item], output_layer)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def get_model(self):
        model = self.model
        return model


