import tensorflow as tf
from tensorflow import keras
from tensorflow import constant_initializer
import numpy as np
from pathlib import Path

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.opt = keras.optimizers.SGD(learning_rate=0.01)
        self.model = keras.models.Sequential([tf.keras.layers.Input((1,), name="inputs"),
                                keras.layers.Dense(1, kernel_initializer=constant_initializer(0.5),
                                                    bias_initializer=keras.initializers.Ones(), name="outputs")
                                ])
        self.model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.SGD(learning_rate=0.01),
              metrics="mae")

    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float32, name="inputs")])
    def __call__(self, x):
        return self.model(x)

    #@tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float32, name="inputs")])
    @tf.function(input_signature=[])
    def my_train(self):
        x=np.array([i for i in range(100)])
        x = x*0.0005
        y=x*2
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)  
            loss_value = tf.reduce_mean(keras.losses.mean_squared_error(y, logits))

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value



model = MyModel()

model_path = "./output"
tf.saved_model.save(model, model_path
            , signatures={
            'serving_default' : 
            model.__call__.get_concrete_function(tf.TensorSpec([None, 1], tf.float32, name="inputs")),
            'my_train' : 
            model.my_train.get_concrete_function()}
            )


print("predict before training : ", model.model.predict([10]))
model.my_train()
print("predict after retraining: ", model.model.predict([10]))
