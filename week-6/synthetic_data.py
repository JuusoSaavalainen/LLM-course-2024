from llama_index.llms.ollama import Ollama
import dspy
import pandas as pd
import random

llm = Ollama(model="llama3", request_timeout=60.0)

# you can use DSPY (https://github.com/stanfordnlp/dspy), but you can also choose another method of interacting with an LLM
dspy.settings.configure(lm=llm)

# Task: implement a method, that will take a query string as input and produce N misspelling variants of the query.
# These variants with typos will be used to test a search engine quality.
# Example
# Query: machine learning applications
# Possible Misspellings:
# "machin learning applications" (missing "e" in "machine")
# "mashine learning applications" (phonetically similar spelling of "machine")
# "machine lerning aplications" (missing "a" in "learning" and "p" in "applications")
# "machin lerning aplications" (combining multiple typos)
# "mahcine learing aplication" (transposed letters in "machine" and typos in "learning" and "applications")
#
# Questions:
# 1. Does the search engine produce the same results for all the variants?
# 2. Do all variants make sense?
# 3. How to improve robustness of the method, for example, skip known abbreviations, like JFK or NBC.
# 4. Can you test multiple LLMs and figure out which one is the best?
# 5. Do the misspellings capture a variety of error types (phonetic, omission, transposition, repetition)?

################################################################################################################
# Solution

# get the queries
queries_df = pd.read_csv('web_search_queries.csv')
queries = queries_df['Query'].tolist()

# print(queries[:5])


def wanted_method(query, n):
    typo_queries = []
    while len(typo_queries) < n:
        words = query.split()
        num_words_to_change = random.choices(
            [1, 2, 3], weights=[0.6, 0.3, 0.1], k=1)[0]
        for i in range(num_words_to_change):
            word_idx = random.randint(0, len(words) - 1)
            if words[word_idx].isupper():
                continue
            words[word_idx] = generate_error_on_single_word(words[word_idx])
        if ' '.join(words) not in typo_queries:
            typo_queries.append(' '.join(words))
        else:
            continue
    return typo_queries


def generate_error_on_single_word(word):
    # possbile error types
    error_types = ['omission',
                   'transposition', 'repetition', 'addition']
    error_type = random.choice(error_types)

    if error_type == 'omission':
        # omission error: remove a character
        char_idx = random.randint(0, len(word) - 1)
        word = word[:char_idx] + word[char_idx + 1:]

    elif error_type == 'transposition':
        # Transpose two adjacent characters
        char_idx = random.randint(0, len(word) - 2)
        word = word[:char_idx] + word[char_idx + 1] + \
            word[char_idx] + word[char_idx + 2:]

    elif error_type == 'repetition':
        # repetition error: repeat a character
        char_idx = random.randint(0, len(word) - 1)
        word = word[:char_idx] + word[char_idx] + \
            word[char_idx] + word[char_idx + 1:]

    elif error_type == 'addition':
        # addition error: add a character
        char_idx = random.randint(0, len(word) - 1)
        word = word[:char_idx] + \
            chr(random.randint(97, 122)) + word[char_idx:]

    return word


for query in queries:
    print(f'Original Query: {query}')
    typo_queries = wanted_method(query, 5)
    print(f'Misspelled Queries: {typo_queries}')
    print('\n')

    # save the misspelled queries in csv column for original query and misspelled queries
    queries_df.loc[queries_df['Query'] == query, 'Misspelled Queries'] = str(
        typo_queries)

queries_df.to_csv('web_search_queries_with_typos.csv', index=False)


################################################################################################################
