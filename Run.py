import numpy as np
from sklearn.utils import shuffle
from Loader import Loader
from Metric import Metric
from model.NeuMF import NeuMF

class Run:

    def __init__(self):

        # data 로드
        loader = Loader()

        print('start data load..')

        num_neg = 4
        uids, iids, self.df_train, self.df_test, \
        self.df_neg, self.users, self.items, item_lookup = loader.load_dataset()
        user_input, item_input, labels = loader.get_train_instances(uids, iids, num_neg, len(self.items))

        print('end data load..')

        # input data 준비
        user_data_shuff, item_data_shuff, label_data_shuff = shuffle(user_input, item_input, labels)
        self.user_data_shuff = np.array(user_data_shuff).reshape(-1,1)
        self.item_data_shuff = np.array(item_data_shuff).reshape(-1,1)
        self.label_data_shuff = np.array(label_data_shuff).reshape(-1,1)

    def run(self):

        nmf = NeuMF(len(self.users), len(self.items))  # Neural Collaborative Filtering
        self.model = nmf.get_model()
        self.model.fit([self.user_data_shuff, self.item_data_shuff], self.label_data_shuff, epochs=20,
                       batch_size=256, verbose=1)

        return self.model

    def calculate_top_k_metric(self):
        metric = Metric()
        hit_lst = metric.evaluate_top_k(self.df_neg, self.df_test, self.model, K=10)
        hit = np.mean(hit_lst)

        return hit

if __name__ == '__main__':

    ncf = Run()
    model = ncf.run()

    # top-k metric
    top_k_metric = ncf.calculate_top_k_metric()
    print('metric:', top_k_metric)

    # user 한 명에 대한 prediction 예시
    user_id = 0
    user_candidate_item = np.array([134, 6783, 27888, 8362, 25]).reshape(-1, 1)
    user_input = np.full(len(user_candidate_item), user_id, dtype='int32').reshape(-1, 1)

    predictions = model.predict([user_input, user_candidate_item])
    predictions = predictions.flatten().tolist()
    item_to_pre_score = {item[0]: pre for item, pre in zip(user_candidate_item, predictions)}  # 후보 아이템 별 예측값
    item_to_pre_score = dict(sorted(item_to_pre_score.items(), key=lambda x: x[1], reverse=True))

    recommend_item_lst = list(item_to_pre_score.keys())
    print('recommend:', recommend_item_lst)
