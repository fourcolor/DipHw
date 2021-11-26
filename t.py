import  tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import  os
import datetime
from tensorflow.python.keras.optimizer_v2 import adam
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# include_top=True，表示會載入完整的 VGG16 模型，包括加在最後3層的卷積層
# include_top=False，表示會載入 VGG16 的模型，不包括加在最後3層的卷積層，通常是取得 Features
# 若下載失敗，請先刪除 c:\<使用者>\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels.h5
model = VGG16(weights='imagenet', include_top=False,input_shape=[32,32,3]) 
for layer in model.layers[:18]:
    layer.trainable = False

# 從頂部移出一層
x = model.output
x = Flatten()(x) # Flatten dimensions to for use in FC layers
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(10, activation='softmax')(x) # Softmax for multiclass
model = Model(inputs=model.input, outputs=x)
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)
# Input：要辨識的影像
def preprocess(x, y):
    # [0~1]
    x = 2*tf.cast(x, dtype=tf.float32) / 255.-1
    y = tf.cast(y, dtype=tf.int32)
    return x,y


(x,y), (x_test, y_test) = datasets.cifar10.load_data()
y = tf.squeeze(y, axis=1)
y = tf.one_hot(y, depth=10)
y_test = tf.squeeze(y_test, axis=1)
y_test = tf.one_hot(y_test, depth=10)
print(x.shape, y.shape, x_test.shape, y_test.shape)


train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model.summary()
model.fit(x=x, 
		y=y, 
		epochs=20, 
		validation_data=(x_test, y_test), 
		callbacks=[tensorboard_callback])
model.save("model.h5")
"""def main():

    # [b, 32, 32, 3] => [b, 1, 1, 512]
    model.summary()
    optimizer = optimizers.Adam(lr=1e-4)

    # [1, 2] + [3, 4] => [1, 2, 3, 4]
    variables = model.trainable_variables

    for epoch in range(20):

        for step, (x,y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                logits = model(x)
                # [b] => [b, 10]
                y_onehot = tf.one_hot(y, depth=10)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step %100 == 0:
                print(epoch, step, 'loss:', float(loss))



        total_num = 0
        total_correct = 0
        for x,y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)
	
	model.save("model.h5")



if __name__ == '__main__':
    main()"""