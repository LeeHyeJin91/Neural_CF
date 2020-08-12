import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class NeuMF:

    def __init__(self, user_num, item_num):

        latent_features = 8

        # Input
        user = Input(shape=(1,), dtype='int32')
        item = Input(shape=(1,), dtype='int32')

        # User embedding for GMF
        gmf_user_embedding = Embedding(user_num, latent_features, input_length=user.shape[1])(user)
        gmf_user_embedding = Flatten()(gmf_user_embedding)

        # Item embedding for GMF
        gmf_item_embedding = Embedding(item_num, latent_features, input_length=item.shape[1])(item)
        gmf_item_embedding = Flatten()(gmf_item_embedding)

        # User embedding for MLP
        mlp_user_embedding = Embedding(user_num, 32, input_length=user.shape[1])(user)
        mlp_user_embedding = Flatten()(mlp_user_embedding)

        # Item embedding for MLP
        mlp_item_embedding = Embedding(item_num, 32, input_length=item.shape[1])(item)
        mlp_item_embedding = Flatten()(mlp_item_embedding)

        # GMF layers
        gmf_mul =  Multiply()([gmf_user_embedding, gmf_item_embedding])

        # MLP layers
        mlp_concat = Concatenate()([mlp_user_embedding, mlp_item_embedding])
        mlp_dropout = Dropout(0.2)(mlp_concat)

        # Layer1
        mlp_layer_1 = Dense(units=64, activation='relu', name='mlp_layer1')(mlp_dropout)  # (64,1)
        mlp_dropout1 = Dropout(rate=0.2, name='dropout1')(mlp_layer_1)                    # (64,1)
        mlp_batch_norm1 = BatchNormalization(name='batch_norm1')(mlp_dropout1)            # (64,1)

        # Layer2
        mlp_layer_2 = Dense(units=32, activation='relu', name='mlp_layer2')(mlp_batch_norm1)  # (32,1)
        mlp_dropout2 = Dropout(rate=0.2, name='dropout2')(mlp_layer_2)                        # (32,1)
        mlp_batch_norm2 = BatchNormalization(name='batch_norm2')(mlp_dropout2)                # (32,1)

        # Layer3
        mlp_layer_3 = Dense(units=16, activation='relu', name='mlp_layer3')(mlp_batch_norm2)  # (16,1)

        # Layer4
        mlp_layer_4 = Dense(units=8, activation='relu', name='mlp_layer4')(mlp_layer_3)       # (8,1)

        # merge GMF + MLP
        merged_vector = tf.keras.layers.concatenate([gmf_mul, mlp_layer_4])

        # Output layer
        output_layer = Dense(1, kernel_initializer='lecun_uniform', name='output_layer')(merged_vector) # 1,1 / h(8,1)초기화

        # Model
        self.model = Model([user, item], output_layer)
        self.model.compile(optimizer= 'adam', loss= 'binary_crossentropy')

    def get_model(self):
        model = self.model
        return model




