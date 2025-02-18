import gensim
from collections import Counter
from wordcloud import WordCloud 

file_path = "Auguste_Maquet.txt"

#load the corpus
with open(file_path, "r", encoding = "utf-8") as file:
    corpus = file.read()

#Tokenize the text using gensim 
tokens = gensim.utils.simple_preprocess(corpus, deacc = True)


#Count word frequencies
word_freq = Counter(tokens)

#Analyze the vocabulary size and coverage
total_tokens = sum(word_freq.values())
vocab = sorted(word_freq.items(), key= lambda x: x[1], reverse = True)


#Calculate the coverage 
coverage = 0
coverage_threshold = 0.95
vocab_size = 0

for word, freq in vocab:
    coverage += freq/total_tokens
    vocab_size += 1

    if coverage >= coverage_threshold:
        break

print(f"Vocabulary size for {coverage_threshold * 100}%coverage:{vocab_size}")

#create a word cloud object

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(corpus)



