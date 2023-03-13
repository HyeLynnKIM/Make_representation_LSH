# 경로 가끔 잘 안먹어서 수동 지정
import sys
sys.path.append("/data/tapas_my/tapas-master")
import pandas as pd
import numpy as np
from sparselsh import LSH
from scipy.sparse import csr_matrix
import json

from tapas.models import table_retriever_model as rm
from tapas.models.bert import modeling
from tapas.models.bert import table_bert
from tapas.utils import text_utils
from tapas.utils import constants

import dataclasses
from typing import Iterable, List, Mapping, Optional, Text, Tuple
from official.nlp.bert import tokenization
import tensorflow.compat.v1 as tf
import tensorflow as tf2


tf.disable_v2_behavior()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# 세션 생성
sess = tf.compat.v1.Session()


dev_query = []
# features = {}

_EMPTY = '[EMPTY]'
_SEP = '[SEP]'
_CLS = '[CLS]'

@dataclasses.dataclass(frozen=True)
class ConversionConfig:
  vocab_file= '/data/tapas_my/tapas_nq_retriever_large/vocab.txt'
  # vocab_file: Text
  max_seq_length= 128
  max_query_length = 128
  # max_column_id: int
  # max_row_id: int
  strip_column_names: True

config = ConversionConfig

def _get_projection_matrix(name, num_columns):
    return tf.get_variable(
        name,
        shape=[256, num_columns],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

def create_int_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

def _get_pieces(tokens):
  return (token.piece for token in tokens)

def _get_type_representation(hidden_representation, prejection):
  return tf.matmul(hidden_representation, prejection, transpose_b=True)

@dataclasses.dataclass(frozen=True)
class Token:
  original_text: Text
  piece: Text

@dataclasses.dataclass
class RetrieverConfig:
  bert_config=modeling.BertConfig(60000)
  max_query_length: int
  use_out_of_core_negatives: bool = False
  mask_repeated_tables: bool = False
  mask_repeated_questions: bool = False
  ignore_table_content: bool = False
  disabled_features: List[Text] = dataclasses.field(default_factory=list)
  use_mined_negatives: bool = False

reconfig = RetrieverConfig

class TapasTokenizer:
    def __init__(self, vocab_file):
        self._basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
        self._wp_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    def get_vocab(self):
        return self._wp_tokenizer.vocab.keys()

    def tokenize(self, text):
        if text_utils.format_text(text) == constants.EMPTY_TEXT:
            return [Token(_EMPTY, _EMPTY)]
        tokens = []
        for token in self._basic_tokenizer.tokenize(text):
            for piece in self._wp_tokenizer.tokenize(token):
                tokens.append(Token(token, piece))
        return tokens

    def convert_tokens_to_ids(self, word_pieces):
        return self._wp_tokenizer.convert_tokens_to_ids(word_pieces)

    def question_encoding_cost(self, question_tokens):
        # Two extra spots of SEP and CLS.
        return len(question_tokens) + 2

class ToTensorflowExampleBase:
  """Base class for converting interactions to TF examples."""
  def __init__(self, config):
      self._max_seq_length = config.max_seq_length
      self._tokenizer = TapasTokenizer(config.vocab_file)

  def _serialize_text(self, question_tokens):
      tokens = []
      tokens.append(Token(_CLS, _CLS))
      for token in question_tokens:
        tokens.append(token)

      return tokens

  def _pad_to_seq_length(self, inputs, override_max_seq_length=None):
    if override_max_seq_length is not None:
      max_seq_length = override_max_seq_length
    else:
      max_seq_length = self._max_seq_length

    while len(inputs) > max_seq_length: inputs.pop()
    while len(inputs) < max_seq_length: inputs.append(0)

  def _to_token_ids(self, tokens):
    return self._tokenizer.convert_tokens_to_ids(_get_pieces(tokens))

# features_tens = tf.zeros([len(dev_query)], dtype=tf.int16)
question_rep = tf.zeros([len(dev_query), 128], dtype=tf.float32)

class my_model(rm.ModelBuilderData):
    def __init__(self):
        self._tokenizer = TapasTokenizer(config.vocab_file)
        self.TT = ToTensorflowExampleBase(config)
        self.features = {'question_input_ids':[], 'question_input_mask':[]}

        # self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.attention_mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.token_type_ids = tf.placeholder(shape=[None, None], dtype=tf.int32)

    def fill_features(self):
        for i in range(len(query_file)):
            question_text = query_file['questions'][i][0]['originalText']
            print(question_text)
            input()

            q_tokens = self._tokenizer.tokenize(question_text)
            q_tokens = self.TT._serialize_text(q_tokens)

            q_tokens.append(Token(_SEP, _SEP))
            q_input_ids = self.TT._to_token_ids(q_tokens)
            self.TT._pad_to_seq_length(q_input_ids)
            q_input_mask = [1] * len(q_tokens)
            self.TT._pad_to_seq_length(q_input_mask)
            print(q_tokens, q_input_ids)

            self.features['question_input_ids'].append(q_input_ids)
            self.features['question_input_mask'].append(q_input_mask)

    def model_fn(self):
        features_query = {
            "input_ids": self.input_ids,
            "input_mask": self.attention_mask,
            "segment_ids": self.token_type_ids,
            "column_ids": self.token_type_ids,
            "row_ids": self.token_type_ids,
            "prev_label_ids": self.token_type_ids,
            "column_ranks": self.token_type_ids,
            "inv_column_ranks": self.token_type_ids,
            "numeric_relations": self.token_type_ids,
        }
        query_model = table_bert.create_model(
            features=features_query,
            # disabled_features=reconfig.disabled_featrues,
            mode='test',
            bert_config=modeling.BertConfig.from_json_file('/data/tapas_my/tapas_nq_retriever_large/bert_config.json')
        )
        query_hidden_representation = query_model.get_pooled_output()
        query_projection = _get_projection_matrix("text_projection", num_columns=query_hidden_representation.shape[1])
        query_rep = _get_type_representation(query_hidden_representation, query_projection)

        return query_rep
############################################################################
## tables.tsv에서 table representation 불러오기
# file_path = '/data/tapas_my/tapas_nq_retriever_large/'
# file_name = 'tables.tsv'

# tsv_file=pd.read_csv(file_path+file_name, delimiter='\t')
#
# table_rep = np.zeros([len(tsv_file), len(tsv_file['table_rep'][0][1:-1].split(','))], dtype=np.float32)
#
# for i in range(len(table_rep)):
#     table_rep[i] = list(map(float, tsv_file['table_rep'][i][1:-1].split(',')))
#
# np.save('large.npy', table_rep)
###################################################################################################
## Query 파일들 불러와서 tapas method를 활용해서 직접 representation으로 변환
query_path = '/data/tapas_my/nq_data_dir/interactions/'
file_name = ['dev.jsonl', 'test.jsonl']

query_file = pd.read_json(f'{query_path}{file_name[1]}', lines=True)
my_model = my_model()

my_model.fill_features()
features = {'question_input_ids':[], "question_input_mask":[]}

query_hidden_representation = np.zeros([len(query_file)+1, 256], dtype=np.float32)
query_hidden_representation[0] = np.arange(256)
# print(len(query_file))
# input()

query_rep = my_model.model_fn()

save_path = '/data/tapas_my/tapas_nq_retriever_large/model.ckpt'

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
saver.restore(sess, save_path)

for i in range(int(len(my_model.features['question_input_ids'])/42)):
    question_input_ids = np.array(my_model.features['question_input_ids'][i*42:(i+1)*42], dtype=np.int32)
    question_input_mask = np.array(my_model.features['question_input_mask'][i*42:(i+1)*42], dtype=np.int32)
    question_token_type_ids = np.zeros_like(question_input_ids)

    query_rep_array = sess.run(query_rep, feed_dict={
        my_model.input_ids: question_input_ids,
        my_model.attention_mask: question_input_mask,
        my_model.token_type_ids: question_token_type_ids
    })
    query_rep_array = np.array(query_rep_array)
    query_hidden_representation[1+i*42:1+(i+1)*42] = query_rep_array
    print('finish turn')

    # print(query_hidden_representation[0])
    # print(query_rep_array.shape)
    # input()
    
np.savetxt('test_query.tsv', query_hidden_representation, delimiter='\t')
print('Finish!')
