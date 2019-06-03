import LSTM_Model

model = LSTM_Model.LSTM_Model()

model.training(epo=50000, continue_training=False, l2_norm=True)

model.evaluation()
