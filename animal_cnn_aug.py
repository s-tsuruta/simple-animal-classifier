from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

#メインの関数定義
def main():
    #numpy形式に保存したファイルの読み込み
    X_train, X_test, y_train, y_test = np.load("./animal_aug.npy")
    #正規化=>RGBは0~255までの256階調で表現されているので、0~1までの数字に変換する（最大値で割る）
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    #one-hot-vector:正解であれば１、他は０
    #[0,1,2]を[1,0,0],[0,1,0],[0,0,1]
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    #モデルのトレーニング関数の呼び出し
    model = model_train(X_train, y_train)
    #モデルの評価関数の呼び出し
    model_eval(model, X_test, y_test)

def model_train(X, y):
    model = Sequential()
    #32個の各フィルタ3x3、padding:畳みこみ結果が同じサイズになるようにピクセルを左右に足す、input_shape:入力データ（画像）の形状
    model.add(Conv2D(32, (3,3), padding = "same", input_shape=X.shape[1:]))
    model.add(Activation('relu')) #活性化関数。正の部分だけ通して、負の部分は０にする
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    #一番大きな値をとりだす
    model.add(MaxPooling2D(pool_size = (2,2)))
    #データの25%を捨てる
    model.add(Dropout(0, 25))

    model.add(Conv2D(64, (3,3), padding = "same"))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0, 25))

    # データを一列に並べる
    model.add(Flatten())
    #dense:全結合層
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    #それぞれの画像が一致する確率を足しこむと結果が１になる
    model.add(Activation('softmax'))
    #素の結果の終了

    #最適化手法 optimizers.rmspop:トレーニング時の更新プログラム、lr:学習レイト、decay:学習率の低下率
    opt = keras.optimizers.rmsprop(lr = 0.0001, decay=1e-6)

    #評価手法の宣言 loss：損失関数、正解と推定値との誤差、,etrics：正答率
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

    model.fit(X, y, batch_size = 32, epochs = 50)

    #モデルの保存
    model.save('./animal_cnn_aug.h5')

    return model


def model_eval(model, X, y):
    #verbose:途中経過を表示する
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])

if __name__ == "__main__":
    main()
