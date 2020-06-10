# NPMT++
Research and development project @ IITB

Steps to run https://github.com/posenhuang/NPMT [[here]](npmt.md)

## Implementation for Neural phrase to phrase MT

Neural Phrase-to-Phrase Machine Translation

Version 1: https://openreview.net/pdf?id=S1gtclSFvr

Version 2: https://arxiv.org/pdf/1811.02172.pdf

## Data

Dataset can be downloaded from here 

http://www.cfilt.iitb.ac.in/~moses/iitb_en_hi_parallel/iitb_corpus_download/dev_test.tgz

http://www.cfilt.iitb.ac.in/~moses/iitb_en_hi_parallel/iitb_corpus_download/parallel.tgz
```
+-- data
|   +-- IITB_len7 (parallel corpus with length truncated to 7)
|   +-- IITB_small
```

## Dependencies

pip install -q torch==1.5.0 torchtext==0.4.0

pip install -U spacy

python -m spacy download en


