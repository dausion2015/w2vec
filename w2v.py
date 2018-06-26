# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile
from datetime import datetime
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
import json
from matplotlib.pylab import mpl
dr = os.path.dirname(os.path.abspath(__file__))


dr_out = os.path.join(dr,'output')
os.mkdirs(dr_out)
dr_log = os.path.join(dr,'log')
os.mkdirs(dr_out)
# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.


#####################################################################################################
start =datetime.now()
print('start running at',start)
# current_path = os.path.dirname(os.path.realpath(sys.argv[0])) #这句话的意思是当前文件所在路径 os.path.realpath是当前文件路径sys.argv[0]是当前脚本

# # parser = argparse.ArgumentParser() #参数解析
# # # parser.add_argument(            #增加参数
# #     '--log_dir',
# #     type=str,
# #     default=os.path.join(current_path, 'log'),#缺省值生成结尾是log/的路径
# #     help='The log directory for TensorBoard summaries.')
# parser.add_argument(
#                   '--filename',
#                     type=str,
#                     default= '',#   os.path.join(current_path,'text.zip'),
#                     help='file dir')
#######################################################################################################
# FLAGS, unparsed = parser.parse_known_args() #将所有参数解析成字典 参数值是value 去掉--的log_dir是key 存于FLAGS中

# Create the directory for TensorBoard variables if there is not.
# if not os.path.exists(FLAGS.log_dir): #若在FLAGS字典中不存在log_dir则 通过os.mkdir建立名为FLAGS.log_dir的文件夹
#   os.makedirs(FLAGS.log_dir)

# # Step 1: Download the data.
# url = 'http://mattmahoney.net/dc/'           #网址存于url字符串


# pylint: disable=redefined-outer-name
# def maybe_download(filename, expected_bytes):
#   """Download a file if not present, and make sure it's the right size."""

#   local_filename = os.path.join(gettempdir(), filename) #在本地创建临时文件 filename gettempdir 属于tempfile模块，返回存储临时文件目录的路径若没有返回系统环境变量中的临时变量路径若在没有返回workspace 
#   if not os.path.exists(local_filename):  #文件如果不存在 os.path.exist()判断括号里的路径是否存在若存在返回true 不存在返回false
#    #路径若不存在 执行 urllib.request.urlretrieve  urlretrieve(url, filename=None, callback回调函数, data=None)urlretrieve是下载url的文件 返回值是文件名和服务器响应的header文件信息
#     # url是下载的运城路径 filename是现在到稳定的路径包括文件名 callback是回调函数 当与服务器联接成功或者下载完指定文件可以执行该函数一般用来显示下载完成进度 data是上传到服务器的数据post 
#     local_filename, _ = urllib.request.urlretrieve(url + filename,   #local_filename是urlretrieve返回值filename，_是服务器返回值header, _ url + filename要下载文件的url  local_filename 下载到本地的路径名
#                                                    local_filename)  #还有一个是urlopen 函数 创建一个类文件的文件（将远端文件）可以read readline readlines fileno
                                                  
                                               
  # statinfo = os.stat(local_filename)  # os.stat是一个；类实例化后 在（）里的指定dir上调用os，stat查看系统信息 这里查看文件大小
  # if statinfo.st_size == expected_bytes: #statinfo.st_size  查看local_filename文件大小是否与expected_bytes相同
  #   print('Found and verified', filename)#与expected_bytes相同 打印 已验证文件名
  # else:
  #   print(statinfo.st_size)              #若不同，用raise  Exception（）抛出异常 异常里可以写提示的文本信息
  #   raise Exception('Failed to verify ' + local_filename +
  #                   '. Can you get to it with a browser?')
  # return local_filename  #返回文件名

# filename = maybe_download('text8.zip', 31344016)   # 执行函数并返回文件名


# Read the data into a list of strings.
filename = '/data/dausion2015/w2vec/sc.zip'
def read_data(filename):                     #创建 zipfile.ZipFile对象 用zipfile.ZipFile处理压缩文件 压缩解压  处理括号里的'zip'
  with zipfile.ZipFile(filename) as f:
    data = list(f.read(f.namelist()[0]).decode(errors='ignore'))   #zipfile.ZipFile.namelist()是将压缩包里的所有文件的文件名组成list namelist()[0]是list、里的第一个文件zipfile.ZipFile.read读取压缩包里的文件
  return data   #把结果作为list返回
########################################################################################
# def read_data(filename):
#   """Extract the first file enclosed in a zip file as a list of words."""
#   with zipfile.ZipFile(filename) as f: #创建 zipfile.ZipFile对象 用zipfile.ZipFile处理压缩文件 压缩解压  处理括号里的'text8.zip'
#     data = tf.compat.as_str(f.read(f.namelist()[0])).split() #zipfile.ZipFile.namelist()是将压缩包里的所有文件的文件名组成list namelist()[0]是list、里的第一个文件zipfile.ZipFile.read读取压缩包里的文件
#     #tf.compat.as_str将str、或者bytes都转换成str，split()括号里制定分割符用制定字符分割，没指定用空格/n /t 等不显示字符分割 这里将多有字符串转换成字符串list
#   return data  #返回压缩包里文件生成的大的单词list 这里是'text8.zip' 语料文件
#############################################################################################################

vocabulary = read_data(filename)  #调用read_data函数将text8.zip 的语料文件生成字符串list存储于vocabulary
# print('55555555555555555555555555555555555',vocabulary[:20])
print('Data size', len(vocabulary)) #打印输出vocabulary长度 其实就是预料文件的size

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 5000  #词频数限制 保留前5000个


def build_dataset(words, n_words):#words是vocabulary n_words 是vocabulary_size
  """Process raw inputs into a dataset."""
  #Counter（计数器）是对字典的补充，用于追踪值的出现次数。统计可迭代对象重复的次数，Counter对象有一个字典接口除了它们在缺失的items时候返回0而不是产生一个KeyError。
  #most_common([n])	从多到少返回一个有前n多的元素的列表，如果n被忽略或者为none，返回所有元素，相同数量的元素次序任意
  #下面对vocabulary 单词进行按照词频进行编码
  count = [['UNK', -1]]  #低频单词设为'UNK'
  count.extend(collections.Counter(words).most_common(n_words - 1))
  #extend是list扩充
  #collections.Counter(words)将单词list统计次品 单词是key出现的次数是value 并且按次数降序排列most_common(n_words - 1))是取Counter(words)前n_words - 1 这里-1是由于在count第一位还有个'UNK', -1
  #这里将单词出现次数的前n_words - 1（49999）位设位高频词汇，将以后的单词作为低频词汇编码位UNK
  dictionary = dict() #建立空字典 对词汇进行编码以便更进一步的onehot编码
  '''
   Counter('abracadabra').most_common(3)
   [('a', 5), ('r', 2), ('b', 2)]
  '''
  #可以看出Counter（iterable）的见过是一些两个元素的元组组成的list元组的第一个元素是词汇list vocabulary中的单词 ，第二个元素是单词出现的次数 所以在和count extend后 每个元素都是一对对的
  #第一个元素是['UNK', -1]
  for word, _ in count: #遍历count 而元素list  每个元素是一对值 第一个元素是单词 存于word
    dictionary[word] = len(dictionary) #字典 dictionary的key是count中的每一个单词 而key是这个字典在没加入这个key时的长度 比如：dict第一个key是UNK 而他还没加入dict中此时长度是0
    # 加第二个key时unk已经加入 dict长度时1所以第二个value时1  所以dictionary的第n个key对应的value时n-
  data = list() #创建空list
  unk_count = 0
  for word in words:  #遍历vocabulary
    index = dictionary.get(word, 0)#dict.get(word, 0) 如果字典有这个key 返回value 没有则返回0  所以这里时 若果词汇表里的word的次品在前 n_words - 1里
    #即在dictionary的key中返回value作为index，若不在 则将0作为索引 即将低频单词归类位UNK
    if index == 0:  # dictionary['UNK'] 若出现一个低频单词归为UNK类则unk_count累加
      unk_count += 1
    data.append(index)#将此时单词生成的索引追加到data 这个list中 这样vocabulary词汇表里的每一个单词都会生成一个编码（0-49999）
    #在data中并且每个单词和它自己的编码会通过各自list的索引对应上 两个list等长
  count[0][1] = unk_count  #count[0]是[['UNK', -1]] count[0][1]是这个-1 将这个-1替换成unk_count的累加值
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys())) #dictionary的key 和value分别是reversed_dictionary的value和key
  #zip能将dictionary.values(), dictionary.keys()了两个list对应位置的元素生成两个元素组成元组的list 在进行dict 将这个元组list转换成字典 key是dictionary.values() 
  # value是 dictionary.keys()
  return data, count, dictionary, reversed_dictionary # 返回这些


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)  #生成data, count, dictionary, reverse_dictionary
dic_dir = os.path.join(dr_out,'dictionary.json')
with open(dic_dir,'w',encoding='utf8') as f:   #json.dumps将python字典对象dumps成字符串 然后通过file。write写进文件保存
    f.write(json.dumps(dictionary))
# with open('E:\\week11w2v\\dictionary.json','w',encoding='utf8') as f:   #json.dumps将python字典对象dumps成字符串 然后通过file。write写进文件保存
#     f.write(json.dumps(dictionary))
redic_dir = os.path.join(dr_out,'reverse_dictionary.json')
with open(redic_dir,'w',encoding='utf8') as f:
    f.write(json.dumps(reverse_dictionary))
# with open('E:\\week11w2v\\reverse_dictionary.json','w',encoding='utf8') as f:
#     f.write(json.dumps(reverse_dictionary))
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])  #第一个['UNK',unk_count] 词频高的前4个
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])  #data[:10]：vocabulary 前10个元素对应的编码  [reverse_dictionary[i] for i in data[:10]]：vocabulary前10个单词

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
#batch_size：一次训练样本的数量, num_skips：上下文和input word生成训练样本的的数量, skip_window：input word左右两边取多少个组成上下文（滑窗）
def generate_batch(batch_size, num_skips, skip_window):
  global data_index  #全局变量 来自函数外
  assert batch_size % num_skips == 0 #batch_size应该是num_skips的整数倍，这样才能保证同一个input word产生的训练样本在一个batch、中
  assert num_skips <= 2 * skip_window #一个inputword和他的context最多能生成2 * skip_window个样本，（能挑出的最大值）训练样本是input word和上下文组成的数值对，所以num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32) #生成128维向量 np.ndarray类似于tf.placeholder 指定一数据类型和shape
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)#生成128*1矩阵
  span = 2 * skip_window + 1  # [ skip_window target skip_window ] 上下文+input word量
  buffer = collections.deque(maxlen=span)  # collections.deque 双端队列  存的是inputword和其上下文编码 最大长度是span 对span是从对面挤出
  if data_index + span > len(data):  #若data_index靠后data_index + span大于data长度 data_index本例中也就是49995 置0 指针回到data开始处
    data_index = 0 
  buffer.extend(data[data_index:data_index + span])#右边入队列span个数据从data中以data_index为起点取出span长度的数剧入队列作为滑窗 data取到data_index + span-1
  data_index += span #更新data_index dataindex在又有基础上+5只是刚开始的时候是这样更新
  for i in range(batch_size // num_skips):#查看一个batch能装的下几个inputword产生的样本 一个批次可以是 多个inputword和其上下文组成的训练样本 遍历不同的inputword
    context_words = [w for w in range(span) if w != skip_window]#当前inputword上下文编码在deque中索引的list样本，从deque队列里产生，中间的inputword 两边是他的上下文inputword位于span的中间位置，他在span中的下标就是skip_window 上下文不包括inputword
    words_to_use = random.sample(context_words, num_skips)#是个list 从上下文随机跳出numskip个准备和inputword组成样本 这个也是编码在deque中的索引上下文不是全都做train 从列表context_words随机挑选num_skips个返回作为train
     #random.sample从itrea中挑选n个作为片段返回
    for j, context_word in enumerate(words_to_use):#words_to_use的长度是numskip个 所以是i * num_skips + j取出buffer的下标  words_to_use是挑出和inputword组成样本的上下本 但这里存的是上下文编码存储在buffer中的下标要想得到真正的内容 需要利用这个下标取buffer里面取
      batch[i * num_skips + j] = buffer[skip_window]#每个i有numskip个j0-numskip-1 batch中同一个inputword出现numskip次 buffer[skip_window]是inputword 由同一个inputword和其numskip个上下文组成样本对  在同一个i内inputword是同一个num_skips各个batch的值都是这个inputword对应的data编码
      labels[i * num_skips + j, 0] = buffer[context_word]#每一个i对应着一个inputword 他出现numskip次 同一个inputword在label中对应numskip个不同的context字# 的labels对应这个inputword上下文的context_word村的data的编码
    if data_index == len(data):#data_index == len(data)dataindex刚超出了data的最大范围1个#若超出了len(data)1个从data的开头从新加载金buffer长度是span
      buffer.extend(data[0:span]) #重置buffer为data开头的前span个
      data_index = span#data_index也被重置为span data[0]-data[pan-1]存储在buffer
    else:
      buffer.append(data[data_index])#若data_index <len(data)buffer右入队data[data_index]左边排出data[data_index-span}
      data_index += 1#data_index自加1 滑窗有交集
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)#若data_index=49995/  span=5 再次执行函数 
  # data_index + span=50000 等于 len(data) 不能执行data_index = 0 但是已超出data范围 等到执行buffer.extend(data[data_index:data_index + span])
  #会出错误 而执行(data_index + len(data) - span)% len(data) 是49990 不出错
  return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1) #生成batch, labels是训练样本一个batch能装4个inputword生成的样本
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], #打印batch.labels里的编码 和字和batch编码对用的单词 label的编码和字
        reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.负样本数量

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)#从0-99里随机选取16个数字组成一个一维array
#np.arrange（valid_window）生成0-99的list 冲里面随机跳出valid_size 16个返回给valid_examples组成list
graph = tf.Graph()

with graph.as_default():
#tf.Graph() 表示实例化一个 tensorflow的图的类
#tf.Graph().as_default() 表示将这个类实例，也就是新生成的图作为整个 tensorflow 运行环境的默认图
#tf.get_default_graph() 来调用（显示这张默认纸）
  # Input data.
  with tf.name_scope('inputs'):
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])#train_inputs时按照data的内容及整个文本集内容编码生成的train tensor 和generate_batch函数中np.ndarray对应上了
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) #
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)#用valid_examples生成1个1*16的tensor常量

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):#目前只能在cpu下运行
    # Look up embeddings for inputs.
    with tf.name_scope('embeddings'):
      embeddings = tf.Variable(         #初始化输入到隐层的权重矩阵 与onehot相乘就是词向量
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) #vocabulary_size 按照高频字顺序生成每一行代表高频字中的一个字是高频字个数50000*128  词向量是128维 初始化重random_uniform初始
      embed = tf.nn.embedding_lookup(embeddings, train_inputs) #tf.nn.embedding_lookup查表embeddings权重矩阵 train_input 原始文本集合编码生成的而编码是根据v高频字来的将他在vocabulary维度上onehot枚以上代表一个字
  #结果是词向量
    # Construct the variables for the NCE loss
    with tf.name_scope('weights'):  #初始化nce loss权重
      nce_weights = tf.Variable(
          tf.truncated_normal(
              [vocabulary_size, embedding_size],
              stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
#necloss
  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  with tf.name_scope('loss'):
    loss = tf.reduce_mean( 
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=vocabulary_size))

  # Add the loss value as a scalar to summary.
  tf.summary.scalar('loss', loss)

  # Construct the SGD optimizer using a learning rate of 1.0.
  with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss) #随机梯度优化器

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)) #将embeddings在猎德方向上对每一个字向量进行归一化处理便于后续求cos相似对权重矩阵 归一化在列上 每一行表示一个词
  normalized_embeddings = embeddings / norm #归一化后为了求余弦相似 方便 模长位1
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,  #valid_dataset是0-99一组数字可以把它看成是reversedict的key 对应着高频字的的字的序号 将他onehot 在embedding
                                            valid_dataset) #得到revdict中对应文字的归一化后的embedding word 用valid_dataset 对normalized_embeddings进行查表操作得到valid_dataset位置对应的词向量
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True) #valid_embeddings还16*embeddingsize  是 normalized_embeddings是 vocabularysize*embeddingsize
#similarity = valid_embeddings还16*embeddingsize ×  embeddingsize*vocabularysize =16*vocabularysize
#similarity的每一行各个分量都代表着16个字中每一个字 与所有vocabularysize个字的cos相似度 顺序还是按照高频字顺序 最大值是1自己和自己的相似度因为 valid_dataset取自高频字数字都在高频字范围内
  # Merge all summaries.
  merged = tf.summary.merge_all()#合并所有op点

  # Add variable initializer.
  init = tf.global_variables_initializer()

  # Create a saver.
  saver = tf.train.Saver()

# Step 5: Begin training.
num_steps = 1000

with tf.Session(graph=graph) as session:
  # Open a writer to write summaries.
  writer = tf.summary.FileWriter(dr_log, session.graph) #写如图

  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in range(num_steps): #产生100001训练 生成100001次数据进行100001次训练
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # Define metadata variable.
    run_metadata = tf.RunMetadata()

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
    # Feed metadata variable to session for visualizing the graph in TensorBoard.
    _, summary, loss_val = session.run(
        [optimizer, merged, loss],
        feed_dict=feed_dict,
        run_metadata=run_metadata)
    average_loss += loss_val  #average_loss初值为0 在for外面 每次循环都生成loss_val在这进行累加

    # Add returned summaries to writer in each step.
    writer.add_summary(summary, step)
    # Add metadata to visualize the graph for the last run.
    if step == (num_steps - 1):#step 0-10000  step == (num_steps - 1)循环到最后一步了
      writer.add_run_metadata(run_metadata, 'step%d' % step)

    if step % 2000 == 0: #每2000step打印average_loss
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0#average_loss清0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0: #每10000step计算相似性
      sim = similarity.eval()#此处相当于 sess.run（similarity）
      for i in range(valid_size):#0-15
        valid_word = reverse_dictionary[valid_examples[i]]#valid_examples每个数字在reverse_dictionary对应的文字取出我觉得这个地方不应该是reverse_dictionary二十高频字以valid_examples中的数字为key在reverse_dictionary中进行选择valuevalid_examples[i]是0-99的数字 reverse_dictionary[valid_examples[i]]是词汇
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]#将每个字对其他所有高频字的相似度牌降序取 1 -top_k + 1 前topk个在高频字中的序号存在neaeest中 list第一个值是1 字和他自己无意义不取用valid_examples选择的词向量和所有词向量计算相似度。由于有自身相似度是1所以排序从四二为开始
        #slim前加-表示相似度是从大到小返回值的前top_k下标，这个下标也是valid_enmbd下标也就是valid_examples下标和词向量val最相似的词向量下标 也就是data的返回将list的从值从小到大的下标返回 (-sim[i, :])是将sim[i, :]从大到小的下表返回
        log_str = 'Nearest to %s:' % valid_word #输出字符初值
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]#k 0-topk-1  nearest存着降序排列的和当前valid_word最相近的topk各字在高频字中的序号也就是reversedict的key
          log_str = '%s %s,' % (log_str, close_word)#利用虚幻拼接字符串
        print(log_str)
  final_embeddings = normalized_embeddings.eval()#相当于sees.run（） 运行后final_embeddings是数值

  # Write corresponding labels for the embeddings.
  with open(dr_log+ '/metadata.tsv', 'w',encoding='utf8',errors='ignore') as f:
    for i in range(vocabulary_size):
      f.write(reverse_dictionary[i] + '\n') #写进文件reverse_dictionary

  # Save the model for checkpoints.
  saver.save(session, os.path.join(dr_log, 'model.ckpt'))#保存检查点

  # Create a configuration for visualizing embeddings with the labels in TensorBoard.
  config = projector.ProjectorConfig()
  embedding_conf = config.embeddings.add()
  embedding_conf.tensor_name = embeddings.name
  embedding_conf.metadata_path = os.path.join(dr_log, 'metadata.tsv')
  projector.visualize_embeddings(writer, config)

writer.close()

# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]#这句始终没理解 low_dim_embs返回几个值
    plt.scatter(x, y,cmap=i/2)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')
  plt.savefig(filename)
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
try:
  # pylint: disable=g-import-not-at-top
  
  from sklearn.manifold import TSNE  #用于降维可实话
  import matplotlib.pyplot as plt

  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact') #用于将维 降到二维便于可视化
  plot_only = 500                                                 #将final_embeddings前500行用tsne降维成2d数据作为坐标embedding数据的生成是按照降序排列高频
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])#字生成的 两个字典点也是按照高频字生成的并且embedding每个字的索引和dict每个字的key或者value相同
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]       #取出reverse_dictionary中和low_dim_embs对应的字 字的顺序和low_dim_embs每行代表的字一样 所以用将为的数据当坐标 在ed空间表示字 但是他和原来额字还是对应上的 在二维图上打印字
  plot_with_labels(low_dim_embs, labels, os.path.join(dr_out, 'tsne.png'))#画图并保存

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
print('********************cost time%s sec'%(datetime.now()-start))

    