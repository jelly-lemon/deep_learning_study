# 注意力机制学习

先安装包：pip install attention

来自项目：https://github.com/philipperemy/keras-attention-mechanism


# 训练结果
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 500, 32)           160000    
_________________________________________________________________
dropout (Dropout)            (None, 500, 32)           0         
_________________________________________________________________
lstm (LSTM)                  (None, 500, 100)          53200     
_________________________________________________________________
last_hidden_state (Lambda)   (None, 100)               0         
_________________________________________________________________
attention_score_vec (Dense)  (None, 500, 100)          10000     
_________________________________________________________________
attention_score (Dot)        (None, 500)               0         
_________________________________________________________________
attention_weight (Activation (None, 500)               0         
_________________________________________________________________
context_vector (Dot)         (None, 100)               0         
_________________________________________________________________
attention_output (Concatenat (None, 200)               0         
_________________________________________________________________
attention_vector (Dense)     (None, 128)               25600     
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense (Dense)                (None, 1)                 129       
=================================================================
Total params: 248,929
Trainable params: 248,929
Non-trainable params: 0
_________________________________________________________________
None
Train on 25000 samples, validate on 25000 samples
Epoch 1/10
25000/25000 [==============================] - 308s 12ms/sample - loss: 0.4666 - accuracy: 0.7668 - val_loss: 0.3003 - val_accuracy: 0.8730
Epoch 2/10
25000/25000 [==============================] - 339s 14ms/sample - loss: 0.2807 - accuracy: 0.8860 - val_loss: 0.2799 - val_accuracy: 0.8856
Epoch 3/10
25000/25000 [==============================] - 335s 13ms/sample - loss: 0.2353 - accuracy: 0.9064 - val_loss: 0.2834 - val_accuracy: 0.8835
Epoch 4/10
25000/25000 [==============================] - 336s 13ms/sample - loss: 0.2159 - accuracy: 0.9150 - val_loss: 0.2889 - val_accuracy: 0.8785
Epoch 5/10
25000/25000 [==============================] - 339s 14ms/sample - loss: 0.1921 - accuracy: 0.9266 - val_loss: 0.3160 - val_accuracy: 0.8746
Epoch 6/10
25000/25000 [==============================] - 337s 13ms/sample - loss: 0.1771 - accuracy: 0.9311 - val_loss: 0.3290 - val_accuracy: 0.8816
Epoch 7/10
25000/25000 [==============================] - 341s 14ms/sample - loss: 0.1595 - accuracy: 0.9380 - val_loss: 0.3271 - val_accuracy: 0.8784
Epoch 8/10
25000/25000 [==============================] - 339s 14ms/sample - loss: 0.1457 - accuracy: 0.9458 - val_loss: 0.3388 - val_accuracy: 0.8789
Epoch 9/10
25000/25000 [==============================] - 339s 14ms/sample - loss: 0.1408 - accuracy: 0.9460 - val_loss: 0.3350 - val_accuracy: 0.8734
Epoch 10/10
25000/25000 [==============================] - 337s 13ms/sample - loss: 0.1270 - accuracy: 0.9502 - val_loss: 0.3788 - val_accuracy: 0.8767
Max Test Accuracy: 88.56 %
Mean Test Accuracy: 87.84 %

Process finished with exit code 0
