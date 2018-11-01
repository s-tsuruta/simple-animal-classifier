from PIL import Image
import os, glob
import numpy as np
#from sklearn import cross_validation
#cross_validation はscikit-learn-0.20からサポートされない
from sklearn import model_selection

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50
num_testdata = 80 #収集した画像の半分（80枚）をテストデータとする
                #残りは回転、反転させてデータ増幅用に使用する

X_test = [] #画像データ
X_train = []
Y_test = [] #ラベルデータ（monkey -> 0, boar -> 1, crow -> 2）
Y_train = []

for index, classlabel in enumerate(classes): #カテゴリごとにファイルを読み込む
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg") #画像をfilesに入れる
    for i, file in enumerate(files):
        if i >= 160: break    #サンプルコードは200枚だが、関連性の低い画像を削除して160枚に絞る
        #pillowのライブラリを使用
        image = Image.open(file) #ファイルを開く
        image = image.convert("RGB") #RGBに変換
        image = image.resize((image_size, image_size)) #サイズを整える
        data = np.asarray(image) #pilowの形式からnumpyの配列形式に変換

        #テストデータとトレイニングデータに分ける
        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            X_train.append(data)
            Y_train.append(index)

            for angle in range(-20, 20, 5): #-20～20まで5度刻みで回転させる
                #回転
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                #反転
                img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)


#X = np.array(X)
#Y = np.array(Y)
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)


#データの分割１：３
#テスト用のデータとトレーニング用のデータに分割
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)

#実行ファイルと同じ階層にnpy形式のファイルを保存
np.save("./animal_aug.npy", xy)
