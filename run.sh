#!/bin/sh
# python mlp/extract_preprocess.py ${1:-null} ${2:-null}
# python mlp/test.py

if [[ $1 == 'extract_preprocess' ]]; then
  python mlp/extract_preprocess.py ${1:-null} ${2:-null} ${3:-null}
elif [[ $1 == 'tf_lstm' ]]; then
  python mlp/rnn_tensorflow.py ${1:-null} ${2:-null} ${3:-null}
fi