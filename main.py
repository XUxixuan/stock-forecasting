#coding=utf-8
"""
使用lstm预测与CDP预测进行对比：
CDP也是通过昨日价格对今日价格进行估计的。
代码是用历史数据对今日最高价进行预测，使用数据集dataset_3.csv

H：昨日最高价　　L：昨日最低价　　C：昨日收盘价
CDP=(H＋L＋C)／3
得到CDP值后，再算今日最高值（AH），今日近高值（NH），今日最低值（AL）及今日近低值（NL）。
PT=H－L   PT为前一天的波幅
今日最高值
AH=CDP＋PT　　PT为前一天的波幅（H－L）
今日近高值
NH=CDP*2－L

今日最低值
AL=CDP－PT
今日近低值
NL=CDP*2－H          
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

rnn_unit=20       #隐层数量
input_size=5
output_size=1
lr=0.0006         #学习率
#——————————————————导入数据——————————————————————
f=open('C:/Users/aspire/Documents/Tencent Files/569273496/FileRecv/股票预测/股票预测/dataset_3.csv')
df=pd.read_csv(f)     #读入股票数据
data=df.iloc[:,1:7].values  
train_end = 2000
time_step = 30

#获取训练集
def get_train_data(batch_size=60,time_step=time_step,train_begin=0,train_end=train_end):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:5]
       y=normalized_train_data[i:i+time_step,5,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y


#获取测试集
def get_test_data(time_step=time_step,test_begin=train_end):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:5]
       y=normalized_test_data[i*time_step:(i+1)*time_step,5]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:5]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,5]).tolist())
    return mean,std,test_x,test_y

# 根据CDP规则来计算最高价
"""
H：昨日最高价　　L：昨日最低价　　C：昨日收盘价
CDP=(H＋L＋C)／3
AH=CDP＋PT　　PT为前一天的波幅（H－L）
"""
def get_test_cdp(test_begin=train_end):
    data_test=data[test_begin:]
    close = data_test[:,3]
    low = data_test[:,2]
    high = data_test[:,1]
    cdp = (high+low+close)/3
    test_cdp = cdp+high-low
    return list(test_cdp)

#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X):
    
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#————————————————训练模型————————————————————
def train_lstm(batch_size=60,time_step=time_step,train_begin=0,train_end=2000):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X)
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print("Number of iterations:",i," loss:",loss_)
        print("model_save: ",saver.save(sess,'C:/Users/aspire/Documents/Tencent Files/569273496/FileRecv/股票预测/股票预测/model_save2/modle.ckpt'))
        #我是在window下跑的，这个地址是存放模型的地方，模型参数文件名为modle.ckpt
        #在Linux下面用 'model_save2/modle.ckpt'
        print("The train has finished")
train_lstm()


#————————————————预测模型————————————————————
def prediction(time_step=time_step):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    # 获取CDP预测结果
    test_cdp = get_test_cdp()
    test_cdp = np.array(test_cdp)

    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)

        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)

        test_y=np.array(test_y)*std[5]+mean[5]
        test_predict=np.array(test_predict)*std[5]+mean[5]

        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差程度
        print("The accuracy of this lstm predict:",acc)

        acc2=np.average(np.abs(test_cdp-test_y[:len(test_cdp)])/test_y[:len(test_cdp)])
        print("The accuracy of CDP predict:",acc2)

        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b',)
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.plot(list(range(len(test_cdp))), test_cdp,  color='g')
        plt.show()

prediction()
