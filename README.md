训练集使用Imagenet数据集，不是一般的大，140G左右。
Imagenet下载下来有训练集和验证集，训练集有1000个压缩文件，代表一千类。
tar.sh 将100个压缩文件加压缩成文件夹
deletetar.sh 将压缩文件夹删除
synset_word.txt是1000类的标签数据
move100.py 创建一个100类的训练集，测试集，不是谁都有能力跑1000类的
cretrain_test.py 创建训练集，测试集的txt文件，主要用于模型训练时读取数据，获取标签方便
torch-alexnet.py/torch-alexnet-3D.py 使用pytorch写的Alexnet模型，训练这100类，3D，代表该模型训练的是4维图片，Vgg同理
