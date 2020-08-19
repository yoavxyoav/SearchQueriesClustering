from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
    return message.replace(' ', '_')


def lda_top_words(queries, n_top_words):
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    # max_features=n_features,
                                    stop_words='english')

    tf = tf_vectorizer.fit_transform(queries)

    lda = LatentDirichletAllocation(n_components=1, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return top_words(lda, tf_feature_names, n_top_words)
