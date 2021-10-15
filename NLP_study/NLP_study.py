import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "hello world python packages hello",
    "hello TODO problems terminal",
    "I love my dog!"
]
# 【提示】即使句子中有标点符号，Tokenizer 也能正常处理
# 【易错点】num_words 在训练分词器阶段不起作用，生成的 word_index
# 会包含所有单词。但是在 texts_to_sequences 时，句子中就只保留出现频率为 num_words-1
# 的单词序号，其余的会被直接删除掉。
# oov_token：用该单词代替不认识的单词
tokenizer = Tokenizer(num_words=100, oov_token="<oov>")
# 训练分词器
tokenizer.fit_on_texts(sentences)
# 得到每个单词的索引（即数字编号）
word_index = tokenizer.word_index
print(word_index)

sentences = [
    "hello world python packages hello",
    "hello TODO problems terminal",
    "I love my dog!",
    "catch my breath"
]
# 句子转序列
# 【注意】若 sentences 中出现了训练集中以外的单词，那么在 texts_to_sequences
# 时会被直接删除！
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
# 使用 pad_sequences 来填充，使各句子长度一致
# 默认在前填充 0
padded = pad_sequences(sequences)
print(padded)