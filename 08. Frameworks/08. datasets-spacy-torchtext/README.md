## 一. spacy

代码可直接全部运行

```python
# pip install spacy
# python -m spacy download en_core_web_sm
import spacy

# 加载预训练的spaCy模型
en_nlp = spacy.load("en_core_web_sm") # en: 英文，core: 通用模型，web: 网络数据，sm: 小型模型
string = "What a lovely day it is today!"
print([token.text for token in en_nlp.tokenizer(string)])
```

## 二. torchtext

代码可直接全部运行

```python
# pip install torch==2.2.0
# pip install torchtext==0.17.0
from torchtext.vocab import build_vocab_from_iterator

# 使用torchtext库构建词汇表
data = [
    ["hello", "world"],
    ["hello", "there"],
    ["goodbye", "world"]
] # 这个data可以使用datasets.load_dataset()加载的数据集
vocab = build_vocab_from_iterator(data, specials=["<unk>", "<pad>", "<bos>", "<eos>"]) 
print(f"vocab size: {len(vocab)}")                  # 词汇表大小
print(f"string to index: {vocab.get_stoi()}")       # 获取string-to-index
print(f"index to string: {vocab.get_itos()}")       # 获取index-to-string
print(f"index of 'hello': {vocab['hello']}")        # 获取"hello"的索引
print(f"index of 'unk': {vocab['<unk>']}")          # 获取"<unk>"的索引
print(f"'aaaa' in vocab: {'aaaa' in vocab}")        # 判断"aaaa"是否在词汇表中
# print(vocab["aaaa"])                              # 获取"aaaa"的索引，会报错
vocab.set_default_index(vocab["<unk>"])             # 设置默认索引为"<unk>", 如果不设置，那么索引到不存在的词时会报错
print(f"'aaaa' index: {vocab['aaaa']}")             # 获取"aaaa"的索引，返回"<unk>"的索引
vocab.lookup_indices(data[0])                       # 获取data[0]中每个词的索引
vocab.lookup_tokens(vocab.lookup_indices(data[0]))  # 获取data[0]中每个索引对应的词
```

## 三. datasets

代码可直接全部运行

```bash

# pip install huggingface_hub datasets

# 使用huggingface_hub库寻找有关数据集，使用国内源加速：export HF_ENDPOINT=https://hf-mirror.com
from huggingface_hub import list_datasets
results = list_datasets(search="chinese english translation") # 寻找英汉翻译相关数据集
for dataset in results:
    print(f"- {dataset.id}")
"""
- Garsa3112/ChineseEnglishTranslationDataset
- Lots-of-LoRAs/task807_pawsx_chinese_english_translation
- Lots-of-LoRAs/task781_pawsx_english_chinese_translation
- supergoose/flan_combined_task781_pawsx_english_chinese_translation...
"""

results = list_datasets(search="object detection") # 寻找目标检测相关数据集
for dataset in results:
    print(f"- {dataset.id}")
"""
- fcakyon/gun-object-detection
- keremberke/football-object-detection
- keremberke/clash-of-clans-object-detection
- keremberke/nfl-object-detection....
"""


# 使用datasets库加载指定数据集
from datasets import load_dataset
dataset = load_dataset("bentrevett/multi30k")
print(dataset)
"""
DatasetDict({
    train: Dataset({
        features: ['en', 'de'],
        num_rows: 29000
    })
    validation: Dataset({
        features: ['en', 'de'],
        num_rows: 1014
    })
    test: Dataset({
        features: ['en', 'de'],
        num_rows: 1000
    })
})
"""

# 获取训练集、验证集和测试集, train_data、val_data和test_data都是迭代器
train_data = dataset["train"]       
val_data = dataset["validation"]
test_data = dataset["test"]
print(train_data[0])
"""
{'en': 'Two young, White males are outside near many bushes.', 
'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}
"""


# 结合spacy将数据集中的数据token化
import spacy
en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_sm")
def tokenize(example, en_nlp, de_nlp):
    en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])]
    de_tokens = [token.text for token in de_nlp.tokenizer(example["de"])]
    return {"en_tokens": en_tokens, "de_tokens": de_tokens}
train_data = train_data.map(tokenize, fn_kwargs={"en_nlp": en_nlp, "de_nlp": de_nlp})
print(train_data[0]) 
"""
{'en': 'Two young, White males are outside near many bushes.', 
'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.', 
'en_tokens': ['Two', 'young', ',', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.'], 
'de_tokens': ['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.']}
"""


# 结合torchtext构建词汇表将数据集中的数据序号化
from torchtext.vocab import build_vocab_from_iterator
unk_token = "<unk>"
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

en_vocab = build_vocab_from_iterator(
    train_data["en_tokens"],
    min_freq=2,
    specials=special_tokens,
)

de_vocab = build_vocab_from_iterator(
    train_data["de_tokens"],
    min_freq=2,
    specials=special_tokens,
)
en_vocab.set_default_index(en_vocab[unk_token]) # 设置默认索引为"<unk>"
de_vocab.set_default_index(de_vocab[unk_token]) # 设置默认索引为"<unk>"

def numericalize_example(example, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    de_ids = de_vocab.lookup_indices(example["de_tokens"])
    return {"en_ids": en_ids, "de_ids": de_ids}
train_data = train_data.map(numericalize_example, fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab})
print(train_data[0])
"""
{'en': 'Two young, White males are outside near many bushes.', 
'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.', 
'en_tokens': ['Two', 'young', ',', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.'], 
'de_tokens': ['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.'], 
'en_ids': [19, 25, 15, 1169, 808, 17, 57, 84, 336, 1339, 5],
'de_ids': [21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4]}
"""

# 将序号化的数据集转换为torch.Tensor
data_type = "torch"
format_columns = ["en_ids", "de_ids"]

train_data = train_data.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)
print(train_data[0])
"""
{'en_ids': tensor([  19,   25,   15, 1169,  808,   17,   57,   84,  336, 1339,    5]), 
'de_ids': tensor([  21,   85,  257,   31,   87,   22,   94,    7,   16,  112, 7910, 3209, 4]), 
'en': 'Two young, White males are outside near many bushes.', 
'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.', 
'en_tokens': ['Two', 'young', ',', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.'], 
'de_tokens': ['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.']}
"""
```

