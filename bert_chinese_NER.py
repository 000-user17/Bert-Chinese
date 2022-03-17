import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt   #jupyter要matplotlib.pyplot
from pytorch_transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import pandas as pd
from pandas import DataFrame
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
device

tokenizer = BertTokenizer.from_pretrained('./bert-pretrained-chinese') #必须要./表示当前文件夹的某个文件
model = BertModel.from_pretrained('./bert-pretrained-chinese')
model.to(device)

train = pd.read_csv('./dataload/train.csv')
test = pd.read_csv('./dataload/test.csv')


'''利用one for one 对train.csv的每句话进行分词，再带入bert得到embedding'''
encoded_layers = []
tokens = []
for l in range(train.shape[0]):  #train.shape[0]表示有多少行数据
    token = [one for one in train['text'][l]]
    tokens.append(token)

    indexed_token = tokenizer.convert_tokens_to_ids(token)
    indexed_token_tensor = torch.tensor([indexed_token]).to(device)

    segments_id = [1] * len(token)
    segments_id_tensor = torch.tensor([segments_id]).to(device)

    model.eval() #eval()将我们的模型置于评估模式，而不是训练模式。在这种情况下，评估模式关闭了训练中使用的dropout正则化。
    with torch.no_grad():
        word_embedding, pooled_output = model(indexed_token_tensor, segments_id_tensor)

    encoded_layers.append(word_embedding)

'''获取bert的embedding dim'''
#batch_size, sequence_len, embedding_size
embedding_dim = encoded_layers[0].shape[2]
#bert的embedding dim

'''得到所有的命名实体的字典，然后根据字典得到train.csv每句话的命名实体字典值'''
def tag_to_tensor(tags, tag_to_ix):
    tag_idx = [tag_to_ix[w] for w in tags]
    return torch.tensor(tag_idx)

tag_to_ix = {"B-BANK": 0, "I-BANK": 1, "B-PRODUCT": 2, "I-PRODUCT": 3, "O": 4, "B-COMMENTS_N": 5, "I-COMMENTS_N": 6, "B-COMMENTS_ADJ": 7, "I-COMMENTS_ADJ": 8}  # Assign each tag with a unique index

tags = [] #tags是一个list，里面存放每个训练数据的文本对应的命名实体转化为字典tensor后的形式
for l in range(train.shape[0]):
    tag = train['BIO_anno'][l].split()  #将每一个BIO命名实体分离
    tag_tensors = tag_to_tensor(tag, tag_to_ix)
    tags.append(tag_tensors.to(device))


#本文利用bert先获得训练数据embedding，再加LSTM的模型
'''
embedding_dim: bert输出的每个字的embedding vector长度
hidden_dim: LSTM的每层隐藏元个数，即经过LSTM转化后的embedding dim
tagset_size:命名实体BIO的总个数，用于分类
num_layers:LSTM层数
'''
class BertLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size, num_layers):  #embedding_dim:bert处理后的字的ebedding size，hidden_dim：LSTM中对字的embedding size，tagset_size:BIO命名实体数量,num_layers:LSTM的层数

        super(BertLSTM,self).__init__()

        self.hidden_dim = hidden_dim #self
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=False, dropout=0.5, bias=True, bidirectional=True)
        #LSTM设置,前两个时必须有的，后面依次时LSTM层数，bias时y=wx+b的b，bidirectional双向LSTM
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        #线性层,要用softmax选择命名实体

    def forward(self, embedding_bert):  #forward表示模型在train中要输入的参数，这里只用输入文本经过bert的embedding表示
        hidden_dim = self.hidden_dim

        lstm_out, _ = self.lstm(embedding_bert.permute(1,0,2))
       #打乱embedding_bert顺序使其符合lstm的输入形式
        out = lstm_out[:,:, [i for i in range(hidden_dim)]]
        #取双向lstm输入的前一半部分,不能lstm.out.shape[2]/2，会改变为float类型
        tag_space = self.hidden2tag(out.view(embedding_bert.shape[1], -1))

        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores

'''定义LSTM模型'''
model_LSTM = BertLSTM(embedding_dim, 256, len(tag_to_ix), 3).to(device)
loss_function = nn.NLLLoss() #和log softmax搭配使用
optimizer = optim.SGD(model_LSTM.parameters(), lr=0.01)

def train_LSTM(model, loss_func, optimizer, tags, encoded_layers, epochs):
    losses = []
    iter = []
    for epoch in range(epochs):
        loss_sum=0
        for l in range(len(encoded_layers)):
            if encoded_layers[l].shape[1] != len(tags[l]):
                continue
            #如果命名实体数量和token数量不同，跳过本句
            model.zero_grad() #新batch训练时将梯度归0，防止梯度累积
            tag_scores = model(encoded_layers[l])
            loss = loss_func(tag_scores, tags[l])
            loss.backward()
            optimizer.step()

            loss_sum+=loss.item()
            
        losses.append(loss_sum)
        iter.append(epoch)
        print("the loss of"+ str(epoch) + "is" + str(loss_sum))
    
    plt.title("loss of epoch per————"+str(loss_func)+ ","+ str(epochs)+ "epochs")
    plt.xlabel("loss per 1")
    plt.ylabel("LOSS")
    plt.plot(iter, losses)
    plt.show()


train_LSTM(model_LSTM, loss_function, optimizer, tags, encoded_layers, 10)

'''把所有的token的bert embedding放入LSTM模型中进行一次正向传播，此时模型和数据都还在device对应的设备上'''
model_LSTM.eval()
tag_scores_all = []
with torch.no_grad():
    for l in range(train.shape[0]):
        temp = model_LSTM(encoded_layers[l])#前向传播一次获得每句话每个字对应的命名实体BIO的softmax分数tensor
        tag_scores_all.append(temp)


'''把train中所有句子的BIO命名实体放入BIO_all中'''
BIO_all=[]
for i in range(len(tag_scores_all)):#对于每一句
    max_indexs = []
    BIO = []
    for j in range(tag_scores_all[i].shape[0]):#对于每一句的每个token
        tag_scores = tag_scores_all[i]
        tag_score_list = list(tag_scores[j])

        max_val = max(tag_scores[j])
        max_index = tag_score_list.index(max_val)

        max_indexs.append(max_index)  #一句话的对应的最大索引
    for index in max_indexs:
        BIO.append(list(tag_to_ix.keys())[index])#把tag_to_ix的键（key）转化为list，然后找到index对应位置的key
    
    BIO_all.append(BIO)

