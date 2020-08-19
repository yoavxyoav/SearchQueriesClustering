import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from collections import Counter
from nltk.corpus import stopwords

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import yaml

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

QUERIES_COL = config['queries_col']


def most_common_words(df, queries_col=QUERIES_COL, n_commons=25, remove_stopwords=True, by='word'):
    """
    checks for the most important [queries | individual words] in the search corpus.

    :param df: Pandas DataFrame of which one column in test search queries
    :param queries_col: the search queries column
    :param n_commons: number of top most common
    :param remove_stopwords: use NLTK to remove english stopwords
    :param by: 'word' returns most common *words*, 'query' returns most common *complete queries*.
    :return: Pandas DataFrame with most common words
    """
    if by == 'word':
        entries = df[queries_col].str.cat(sep=' ')
        entries = entries.split(' ')
    elif by == 'query':
        entries = df[queries_col]
    else:
        raise Exception ('The keyword argument "separate" should either be set to "word" or "query"')

    c = Counter(entries)
    commons = c.most_common(n_commons)
    commons = pd.DataFrame(commons)

    if remove_stopwords:
        sw_ind = commons[0].apply(lambda word: word in stopwords.words('english'))
        ind_to_drop = commons[sw_ind].index
        commons.drop(index=ind_to_drop, inplace=True)

    commons.rename(columns={0: "word", 1: "count"}, inplace=True)

    return commons


def plot_stats(df, queries_col=QUERIES_COL, **mcw_kwargs):
    """
    plots graphic information about individual words in the search query corpus.

    :param df: Pandas DataFrame
    :param queries_col: the column that holds the queries in df
    :param top_n: top n words to show
    :param separate: if True, returns most common *words*. Otherwise returns most common *complete queries*.
    :return: None
    """
    commons_df = most_common_words(df, queries_col, **mcw_kwargs)
    top_n = mcw_kwargs['n_commons']

    total = commons_df['count'].sum(axis=0)
    commons_df['freq'] = commons_df['count'].apply(lambda x: x / total)
    if len(commons_df) > top_n:
        commons_df = commons_df[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(18, 4))

    # plotting the word cloud
    wordcloud = WordCloud()
    commons_dict = dict(zip(commons_df['word'], commons_df['freq']))
    wordcloud.generate_from_frequencies(frequencies=commons_dict)
    axes[0].imshow(wordcloud, interpolation="bilinear")
    axes[0].axis("off")

    # plotting the bar count
    words = commons_df['word']
    counts = commons_df['count']
    sns.barplot(x=words, y=counts, ax=axes[1])
    axes[1].tick_params(axis='x', labelrotation=90)

    plt.suptitle(f'Most Common {top_n} {"Words" if mcw_kwargs["by"] == "word" else "Queries"}')

    plt.show()

    return None


def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 50), dtype='f')
    word_labels = [word]
    color_list = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    # gets list of most similar words
    close_words = model.wv.most_similar([word])

    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=19).fit_transform(arrays)

    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                  }
                     )

    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p1.text(df["x"][line],
                df['y'][line],
                '  ' + df["words"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='medium',
                color=df['color'][line],
                weight='normal'
                ).set_size(15)

    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

    plt.title('t-SNE visualization for {}'.format(word.title()))
