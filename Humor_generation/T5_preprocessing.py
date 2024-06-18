import math
import random
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from SoundsLike.SoundsLike import Search
from Phyme import Phyme

# 단어 교체(Word Replacement)
spell = SpellChecker()
stop_words = set(stopwords.words('english'))

def get_word_replacement(sentence):
    cnt = 0
    candidate_word = {}
    while True:
        for origin_word in word_tokenize(sentence):
            if origin_word not in stop_words:
                word = origin_word
                # find those words that may be misspelled
                word = word.replace(word[random.randrange(0, len(word))], '')
                word_dict = {}

                candid = spell.candidates(word)
                if candid:
                    # Get a list of `likely` options
                    for cand in candid:
                        word_dict[cand] = spell.word_usage_frequency(cand)
                    likely = min(word_dict.items())
                    candidate_word[likely[0]] = (likely[1], origin_word)

        if candidate_word:
            minimum = min(candidate_word)
            return sentence.replace(candidate_word[minimum][1], minimum)
        else:
            if cnt >= 3:
                return sentence
            cnt += 1

# 유사 발음(Similar Sound)
def get_similar_sound(sentence):
    candidate_word = {}
    m_candidate_word = {}
    for word in word_tokenize(sentence):
        sim_words = []
        if word not in stop_words:
            try:
                sim_words = Search.endRhymes(word, match_syllables=True, match_alpha=True, generate=False)
                random.shuffle(sim_words)
                if sim_words:
                    for sim_word in sim_words:
                        if sim_word.lower() != word.lower():
                            m_candidate_word[word] = sim_word.lower()
            except:
                pass
            
            else:
                try:
                    sim_words = Search.endRhymes(word, match_syllables=True, match_alpha=False, generate=False)
                    sim_word = sim_words[random.randrange(0, len(sim_words))]
                    candidate_word[word] = sim_word
                except:
                    pass
                
    if m_candidate_word:
        for mcw in m_candidate_word.keys():
            return sentence.replace(mcw, m_candidate_word[mcw])
    elif candidate_word:
        for cw in candidate_word.keys():
            return sentence.replace(cw, candidate_word[cw])
    else:
        return sentence

# 유사 음절(Similar Syllable)
ph = Phyme()

def get_similar_syllable(sentence):
    candidate_word = {}
    for word in word_tokenize(sentence):
        # find rhymes with the same vowels and some of the same consonants, with some swapped out for other consonants. FACTOR -> FASTER
        try:
            rhymes = ph.get_substitution_rhymes(word)
            for rhyme in rhymes.items():
                for r in rhyme[1]:
                    if word.lower() in r.lower():
                        if word.lower() not in stop_words and word.lower() != r.lower():
                            candidate_word[word] = r
        except:
            pass
    
    if candidate_word:
        for cw in candidate_word.keys():
            return sentence.replace(cw, candidate_word[cw])
    else:
        return sentence

# Data Preprocessing
df = pd.read_csv("popular_movie_titles.csv")
df = df.sort_values(by=['votes'], ascending=False)

def get_humor_title(f, col):
    humor_title = []
    cnt = 0
    for i in range(len(df)):   
        if col.iloc[i].lower() != f(col.iloc[i]).lower():
            humor_title.append((col.iloc[i], f(col.iloc[i])))
            cnt += 1
        if cnt >= 63:
            return humor_title
    return humor_title

# 유사 발음과 유사 음절 타이틀 생성
ssd = get_humor_title(get_similar_sound, df['movie title'])
ssd16 = ssd[:16]
ssd32 = ssd[:32]
ssd64 = ssd[:64]

sse = get_humor_title(get_similar_syllable, df['movie title'])
sse16 = sse[:16]
sse32 = sse[:32]
sse64 = sse[:64]

# 결과 출력
print("Similar Sound Titles (Top 16):", ssd16)
print("Similar Sound Titles (Top 32):", ssd32)
print("Similar Sound Titles (Top 64):", ssd64)
print("Similar Syllable Titles (Top 16):", sse16)
print("Similar Syllable Titles (Top 32):", sse32)
print("Similar Syllable Titles (Top 64):", sse64)
