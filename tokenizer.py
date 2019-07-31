#Basic tokenizing of the text corpus..Beginning of the NLP section.
#tokenises makes the vocabulary dictionary and constructs the encoding array

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences=[
	'coding rocks',
	'i am a boy',
	'i am a good boy',
	'i am someone who likes coding',
	'i am a person who is into computing and am very passionate about it'
]


tokenizer=Tokenizer(num_words=50,oov_token="<oov>")
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index

sequences=tokenizer.texts_to_sequences(sentences)

pad_sequences=pad_sequences(sequences,padding='post',maxlen=9)

print(word_index)
print(sequences)
print(pad_sequences)
