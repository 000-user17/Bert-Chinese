# Bert-Chinese
利用Bert+LSTM解决NLP的NER（命名实体识别）————以及SA（情感预测）

#版本：
Python 3.7  pytorch1.6

#bert-pretrained-chinese文件中bert中文与训练模型太大，上传不进来，需要的朋友去https://huggingface.co/bert-base-chinese下载 pytorch_model.bin
另外的vocab.txt和config已经有了

#NER（命名实体识别）
train.csv 共6022条数据text，BIO命名实体，以及情感标签（某一行BIO数量和token数量不一致，因此需要筛查）
test.csv(因为没有标签就没用到)
bert_chinese.ipynb包含我练习尝试pytorch bert的所有过程
bert_chinese.py是精简版本，直接运行就可以得到命名实体识别的结果，存放在BIO_all中，一共6022条

# 方法：
# 1.直接把text token后，因为bert的tokenize是把中文单字和英文单词划为一个token，而train.csv中一个字母如'a'也对应一个命名实体，所以我没用bert的tokenize，直接one for one进行tokenize，然后喂到bert进行embedding。再把bert得到的embedding送到Bi——LSTM里面进行NER任务，训练只训练了双向LSTM

# 2.联合训练bert和LSTM
