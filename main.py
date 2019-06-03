import LSTM_Model

model = LSTM_Model.LSTM_Model()

#모델을 학습하고자 하는 경우
#model.training(epo=50000, continue_training=True, l2_norm=True)

#모델을 평가하고자 하는 경우
#model.evaluation()