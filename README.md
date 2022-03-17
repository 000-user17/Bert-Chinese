# Bert-Chinese
利用Bert+LSTM解决NLP的NER（命名实体识别）————以及SA（情感预测）

#版本：
Python 3.7  pytorch1.6

#bert-pretrained-chinese文件中bert中文与训练模型太大，上传不进来，需要的朋友去https://huggingface.co/bert-base-chinese下载 pytorch_model.bin
另外的vocab.txt和config已经有了

#NER（命名实体识别）
bert_chinese.ipynb包含我练习尝试pytorch bert的所有过程
bert_chinese.py是精简版本，直接运行就可以得到命名实体识别的结果，存放在ans_all中
