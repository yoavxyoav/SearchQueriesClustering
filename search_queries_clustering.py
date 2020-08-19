import pandas as pd
import numpy as np

import lda
import eda_utils
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AffinityPropagation
import hdbscan

import os
import pickle
import time
import datetime
import inspect
import termcolor
import operator
from collections import OrderedDict

import re
from gensim.parsing.preprocessing import strip_punctuation, remove_stopwords, strip_short, strip_tags
from gensim.parsing.preprocessing import preprocess_string
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from collections import Counter
from pprint import pprint

import yaml

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if config['global']['dataframes_prefs']:
    pd.set_option("display.max_columns", config['global']['max_cols'])
    pd.set_option("display.max_rows", config['global']['max_rows'])


class SearchQueriesClustering:
    """
    Class for NLP and Clustering of search queries
    """

    ####################
    ## Initialization ##
    ####################

    def __init__(self, data, queries_col=config['queries_col']):
        """
        Initialized a SearchQueriesClustering instance.

        :param data: a Pandas DataFrame of which one column is search queries or a csv file name with such column
        :param queries_col: The column in df that holds the queries
        """
        self.csv_file_name = None
        self.data = data
        self.current_df_date = ''
        self.df = self._loader()
        self.queries_col = queries_col
        self.doc2vec_model = None
        self.similarity_arr = None
        self.doc2vec_arr = None
        self.uniques_search_queries = None
        self.tokenized_queries = None
        self.clustering_model = None
        self.results = {}
        self.daily_dfs = None
        self.daily_clusters = dict()
        self.daily_scores = None
        self.daily_scores_table = None
        self.all_cluster_names = None

    def _loader(self):
        """
        A helper function for versatile data loading. Enables to provide the init method either filename or DataFrame.
        If provided df argument is a string, loads a csv from file to df,
        if provided df argument is a Pandas DataFrame, loads it to df.
        """
        print('Starting the loader')
        data = self.data

        # caller = inspect.currentframe().f_back.f_locals['self']
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        caller = calframe[2][3]

        if isinstance(data, pd.DataFrame):
            if not caller == 'make_daily_clusters':
                self.csv_file_name = '__session_dataframe__'
            else:
                self.csv_file_name = 'temp_csv_filename'
            return data
        elif isinstance(data, str):
            self.csv_file_name = data
            return self.load_data(data)
        else:
            raise Exception('Wrong type for data provided to SearchQueriesClustering() loader.'
                            'Your options are (1) a csv file (2) a Pandas DataFrame')

    @staticmethod
    def load_data(csv_file, remove_na=True):
        """
        Loads a csv file into a Pandas DataFrame.

        :param csv_file: a csv file with a search queries column
        :param remove_na: removes all rows containing a NaN value in any column
        :return: Pandas DataFrame
        """
        print(f'Loading data from csv file --- {csv_file}')
        csv_file = config['global']['data_dir'] + csv_file
        df = pd.read_csv(csv_file)
        if isinstance(config['global']['limit_dataframe'], int):
            df = df.sample(config['global']['limit_dataframe'])
        if remove_na:
            df = SearchQueriesClustering.remove_na(df)

        if config['global']['always_plot']:
            SearchQueriesClustering.plot_stats(by='word', n_commons=40)
            SearchQueriesClustering.plot_stats(by='query', n_commons=40)

        return df

    @staticmethod
    def drastic_cleanup(remove_models=True, remove_pickles=True):
        """
        Removes pre-saves files from working directory

        :param remove_models: whether to remove all *.model files (usually doc2vec)
        :param remove_pickles: whether to remove all *.pickle files (usually clustering)
        """

        if remove_pickles:
            for item in os.listdir('./'):
                if item.endswith(".pickle"):
                    os.remove(os.path.join('./', item))

        if remove_models:
            for item in os.listdir('./'):
                if item.endswith(".model"):
                    os.remove(os.path.join('./', item))

    ####################
    ## Pre Processing ##
    ####################

    def aggregate_df(self, sum_=config['agg']['sum'], mean_=config['agg']['mean'], first_=config['agg']['first']):
        print(f'Aggregating DataFrame', end=' ... ')
        df = self.df
        col = self.queries_col

        df_sum = df.groupby(by=col).sum()[sum_]
        df_mean = df.groupby(by=col).mean()[mean_]
        df_first = df.groupby(by=col).first()[first_]
        df = pd.concat([df_sum, df_mean, df_first], axis=1).reset_index()
        self.df = df
        print('done')

    @staticmethod
    def remove_na(df):
        """
        Removes undesired samples containing NAs.

        :param df: Pandas Dataframe
        :return: a clean DataFrame
        """
        df
        len_before = len(df)
        ind = df[df.isnull().any(axis=1)].index
        df = df.drop(index=ind)
        len_after = len(df)
        print(f'NaN handling: {len_after} samples left out of {len_before} ({len_before - len_after} removed).')

        return df

    def remove_i_will(self):
        """
        Removes queries that start with "I will"
        """
        i_will_df = self.df[self.queries_col].str.lower().str.startswith('i will')
        proportion = np.round(len(i_will_df[i_will_df]) / len(self.df), 4)
        print(f'Rate of "I Will..." queries to be removed: {proportion}', end=' ... ')
        i_will_inds = i_will_df[i_will_df].index
        self.df.drop(index=i_will_inds, inplace=True)
        print('done')

    @staticmethod
    def clean_search_query(query):
        """
        cleans a query from unwanted characters and lowers the case.

        :param query: search query (str type)
        :return: a clean string
        """
        # removing punctuation
        query = re.sub('[^A-za-z0-9]+', ' ', query)

        # removing spaces from sides of text
        query = query.strip()

        # lower-casing
        query = query.lower()

        return query

    def clean_search_queries(self, clean=True, remove_numeric=True):
        """
        Cleans all search queries in the instance DataFrame

        :param clean: preforms basic pre-processing
        :param remove_numeric: removes queries that purely consist of numbers
        """

        print('Cleaning queries', end=' ... ')
        if clean:
            self.df[self.queries_col] = self.df[self.queries_col].apply(self.clean_search_query)

        if remove_numeric:
            inds_numeric = self.df[self.df[self.queries_col].str.isnumeric()].index
            self.df.drop(index=inds_numeric, inplace=True)
        print('done')

    ############
    ## TF-IDF ##
    ############

    def create_tfidf_similarity(self, queries_series, max_samples=config['tf_idf']['max_samples']):
        """
        Creates a similarity matrix based on TF-IDF of a corpus

        :param queries_series: Pandas Series with search queries
        :param max_samples: the limit for number of samples to process
        :return: similarity array of type Numpy
        """

        print('Creating TF-IDF similarity matrix')
        unique_search_queries = queries_series.value_counts().index.tolist()
        unique_search_queries = unique_search_queries[:max_samples]
        tfidf_unique = TfidfVectorizer(stop_words="english", strip_accents='unicode')
        tfidf_unique = tfidf_unique.fit_transform(unique_search_queries)
        tfidf_unique = tfidf_unique[:max_samples, :]
        similarity_arr = tfidf_unique * tfidf_unique.T
        self.similarity_arr = similarity_arr

    def find_unique_queries(self):
        """
        Finds unique search queries and saves them as a class attribute
        """
        print('Looking for unique search queries')
        self.uniques_search_queries = set(self.df[self.queries_col])

    def show_n_similar_queries(self, query, n_similar):
        """
        Finds the top n_similar sentences from the tf-idf similarity matrix to a given query

        :param query: the search query of which similar queries you wish to find
        :param n_similar: return only n top queries
        :returns: topmost n similar queries
        """

        if self.uniques_search_queries is None:
            self.find_unique_queries()

        # getting the index number of the query from uniques
        ind = self.uniques_search_queries.index(query)

        # getting top n_similar similars
        row = self.similarity_arrarr[ind]
        inds = np.argsort(row)
        n_inds = inds[-n_similar:]
        filtered = np.array(self.uniques_search_queries)[n_inds]

        # getting their similarity scores
        scores = row[n_inds].round(2)

        # combining the simlar queries with their score (for conveneint siplay)
        combined = list(zip(filtered, scores))

        # reversing
        combined = sorted(combined, reverse=True, key=lambda x: x[1])

        if config['global']['verbose']:
            print(f'Query "{query}" (index {ind}) ~~~> {combined}')

        return combined

    def find_all_non_zero_score_similar_queries(self, query):
        """
        Finds the top n_similar sentences from arr to a given sentence

        :param query: the search query of which similar queries we would like to find
        :returns: all non zero score similar queries
        """

        # getting the index number of the query from uniques
        ind = self.uniques_search_queries.index(query)

        # getting the row that represents the
        row = self.similarity_arr[ind]

        # getting all non-zeros
        mask = row != 0
        uniques = np.array(self.uniques_search_queries)[mask]

        # getting their similarity scores
        scores = row[mask].round(2)

        # combining the simlar queries with their score (for conveneint siplay)
        combined = list(zip(uniques, scores))
        combined = sorted(combined, reverse=True, key=lambda x: x[1])

        print(f'Query "{query}" (index {ind}) ~~~> {combined}')

        return combined

    #############
    ## Doc2Vec ##
    #############

    def tokenize_queries(self, pre_process=True):
        """
        Tokenizes the search queries column in the instance DataFrame

        :param pre_process: perform further pre-processing
        """

        print('\nTokenizing', end=' ... ')
        FILTERS = []
        if pre_process:
            FILTERS = [strip_punctuation, remove_stopwords, strip_short, strip_tags]

        self.tokenized_queries = self.df[self.queries_col].apply(lambda s: preprocess_string(s, FILTERS)).tolist()
        print('done')

    def create_tagged_documents(self,
                                tag_sequential=True,
                                tagging_features=config['doc2vec']['tagged_documents_features']):
        """
        Creates a TaggedDocument object which is needed to create a doc2vec model.
        Prior to the taggings, also tokenizes the queries if not already done.

        :param tag_sequential: whether to create sequential tagging or not. Must be `True` if tagging_feature is empty.
        :param tagging_features: additional features to create tagging by (set in YAML config!)
        :return: Gensim tagged documents object
        """

        if self.tokenized_queries is None:
            self.tokenize_queries()

        print('Tagging Documents', end=' ... ')
        for i, list_of_words in enumerate(self.tokenized_queries):
            tag = []
            if tag_sequential:
                tag.append(i)

            for feature in tagging_features:
                tag.append(self.df.iloc[i][feature])

            yield TaggedDocument(list_of_words, tag)
        print('done')

    def fit_doc2vec(self,
                    filename=None,
                    model_params=config['doc2vec']['def_params']):
        """
        Creates a doc2vec embedding.

        :param filename: saves the model to this file name
        :param model_params: doc2vec parameters. Change the values in config.yaml
        :return: Gensim doc2vec object
        """

        tagged_documents = list(self.create_tagged_documents())
        print('Crating a Doc2Vec instance', end=' ... ')
        doc2vec_model = Doc2Vec(**model_params)
        print('done')

        print('Building vocabulary', end=' ... ')
        doc2vec_model.build_vocab(tagged_documents)
        print('done')

        if filename is None:
            model_dir = config['global']['saved_models_dir']
            filename = model_dir + self.csv_file_name + self.current_df_date + repr(model_params) + config['doc2vec'][
                'model_filename_postfix']

        print('Training doc2vec model', end=' ... ')
        start = time.time()
        doc2vec_model.train(tagged_documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
        running_time = (time.time() - start)
        minutes, seconds = divmod(running_time, 60)
        print(f'done (Execution time {int(minutes)} minutes and {seconds:.2f} seconds)')

        if not os.path.isfile(filename) or config['global']['always_save_files'] is True:
            print(f'Saving doc2vec model to {filename}')
            doc2vec_model.save(filename)

        self.doc2vec_model = doc2vec_model
        self.doc2vec_arr = doc2vec_model.docvecs.vectors_docs

    def load_doc2vec_model(self, filename=None):
        """
        Loads a doc2vec model from file.

        :param filename: path to file
        :return: Doc2Vec object
        """

        if filename is None:
            model_dir = config['global']['saved_models_dir']
            filename = model_dir + self.csv_file_name + self.current_df_date + config['doc2vec'][
                'model_filename_postfix']

        print('\t---------------------------------------------------------------------------------- ')
        print(termcolor.colored('\tNOTE: doc2vec model from the provided csv already exists on file and will be used!',
                                'red'))
        print('\tIf you wish to retrain, delete the file or perform the fitting process manually. ')
        print(f'\tLoading doc2vec model from {filename}')
        print('\t---------------------------------------------------------------------------------- ')

        self.doc2vec_model = Doc2Vec.load(filename)
        self.doc2vec_arr = self.doc2vec_model.docvecs.vectors_docs

    @staticmethod
    def marry_queries_to_labels(queries, cluster_labels):
        """
        Helper function to marry queries and cluster labels to form a DataFrame.

        :param queries: queries Pandas Series
        :param cluster_labels: labels Pandas Series
        :return: Pandas DataFrame
        """
        print('Marrying queries to their cluster label')

        labeled_queries = pd.DataFrame(zip(queries, cluster_labels))
        labeled_queries.rename(columns={0: 'query', 1: 'cluster_label'}, inplace=True)
        return labeled_queries

    def assert_doc2vec_model_exists(self):
        """
        Helper function that checks whether instance holds a doc2vec model and array or not.
        """
        assert self.doc2vec_arr is not None and self.doc2vec_model is not None, \
            '''Cannot run clustering without first fitting a doc2vec model. 
                Use wrapper function make_clusters() or create manually with fit_doc2vec()'''

    ################
    ## Clustering ##
    ################

    def k_means_clustering(self,
                           k=config['k_means']['def_params']['k'],
                           load_file=True,
                           **kwargs):
        """
        Fits and predicts a K-Means clustering algorithm.

        :param k: number of clusters
        :param load_file: tries to first load a file instead of fitting if load_file=True
        :return: tuple of (1) Pandas DataFrame with queries and their label (k-th cluster) and (2) the K-Means model
        """

        self.assert_doc2vec_model_exists()

        model_dir = config['global']['saved_models_dir']
        filename = f'{model_dir}_{self.csv_file_name}_{self.current_df_date}_kmeans_{k}_{kwargs}.pickle'

        if load_file and os.path.isfile(filename) and self.csv_file_name != '__session_dataframe__':
            print(f'\t---------------------------------------------------------------------------')
            print(termcolor.colored(f'\tPickle file found! \nLoading from {filename}', 'red'))
            print(f'\t---------------------------------------------------------------------------')
            with open(filename, 'rb') as file:
                k_means = pickle.load(file)
        else:
            k_means = KMeans(n_clusters=k, **kwargs)
            k_means.fit(self.doc2vec_arr)
            with open(filename, 'wb') as file:
                pickle.dump(k_means, file)

        self.clustering_model = k_means

        cluster_labels = k_means.predict(self.doc2vec_arr)
        labeled_queries = self.marry_queries_to_labels(self.df[self.queries_col], cluster_labels)

        return labeled_queries, k_means

    def aff_prop_clustering(self,
                            damping=config['aff_prop']['def_params']['damping'],
                            affinity=config['aff_prop']['def_params']['affinity'],
                            load_file=True,
                            **kwargs):
        """
        Performs Affinity Propagation clustering

        :param damping: damping parameter
        :param affinity: affinity parameter
        :param load_file: tries to first load a file instead of fitting if load_file=True
        :param kwargs: more affinity propagation keyword arguments
        :return: tuple of (1) Pandas DataFrame with queries and their label (k-th cluster) and (2) the Aff Prop model
        """

        self.assert_doc2vec_model_exists()

        limit = config['aff_prop']['limit']
        model_dir = config['global']['saved_models_dir']
        filename = f'{model_dir}_{self.csv_file_name}_{self.current_df_date}_aff_prop_{damping}_{affinity}_{kwargs}.pickle'

        if load_file and os.path.isfile(filename) and self.csv_file_name != '__session_dataframe__':
            print(f'\t---------------------------------------------------------------------------')
            print(termcolor.colored(f'\tPickle file found! \nLoading from {filename}', 'red'))
            print(f'\t---------------------------------------------------------------------------')
            with open(filename, 'rb') as file:
                af = pickle.load(file)
        else:
            aff_prop = AffinityPropagation(affinity=affinity, damping=damping, **kwargs)
            af = aff_prop.fit(self.doc2vec_arr[:limit])
            with open(filename, 'wb') as file:
                pickle.dump(af, file)

        self.clustering_model = aff_prop

        cluster_labels = af.labels_
        labeled_queries = self.marry_queries_to_labels(self.df[self.queries_col], cluster_labels)

        return labeled_queries, af

    def hdbscan_clustering(self,
                           cluster_selection_epsilon=config['hdbscan']['def_params']['cluster_selection_epsilon'],
                           min_samples=config['hdbscan']['def_params']['min_samples'],
                           min_cluster_size=config['hdbscan']['def_params']['min_cluster_size'],
                           allow_single_cluster=config['hdbscan']['def_params']['allow_single_cluster'],
                           cluster_selection_method=config['hdbscan']['def_params']['cluster_selection_method'],
                           leaf_size=config['hdbscan']['def_params']['leaf_size'],
                           load_file=config['global']['always_load_files'],
                           **kwargs):
        """
        Performs HDBScan clustering

        :param cluster_selection_epsilon: cluster_selection_epsilon parameter
        :param min_samples: min_samples parameter
        :param load_file: tries to first load a file instead of fitting if load_file=True
        :param kwargs: more affinity propagation keyword arguments
        :return: tuple of (1) Pandas DataFrame with queries and their label (k-th cluster) and (2) the HDBScan model
        """

        self.assert_doc2vec_model_exists()

        model_dir = config['global']['saved_models_dir']
        filename = f'{model_dir}_{self.csv_file_name}_{self.current_df_date}_hdbscan_{cluster_selection_epsilon}_{min_samples}_{kwargs}.pickle'

        if load_file and os.path.isfile(filename) and self.csv_file_name != '__session_dataframe__':
            print(f'\t---------------------------------------------------------------------------')
            print(termcolor.colored(f'\tPickle file found! \nLoading from {filename}', 'red'))
            print(f'\t---------------------------------------------------------------------------')
            with open(filename, 'rb') as f:
                hdbs = pickle.load(f)
        else:
            hdbs = hdbscan.HDBSCAN(cluster_selection_epsilon=cluster_selection_epsilon,
                                   min_samples=min_samples,
                                   min_cluster_size=min_cluster_size,
                                   allow_single_cluster=allow_single_cluster,
                                   cluster_selection_method=cluster_selection_method,
                                   leaf_size=leaf_size,
                                   **kwargs)

            limit = config['hdbscan']['limit']
            if not isinstance(limit, int):
                limit = -1
            hdbs.fit(self.doc2vec_arr[:limit])

            with open(filename, 'wb') as file:
                pickle.dump(hdbs, file)

        self.clustering_model = hdbs

        cluster_labels = hdbs.labels_
        labeled_queries = self.marry_queries_to_labels(self.df[self.queries_col][:limit], cluster_labels)

        return labeled_queries, hdbs

    def make_clusters(self,
                      algo='hdbscan',
                      save_results=config['global']['always_save_files'],
                      **kwargs):
        """
        Wrapper function to do a clustering of choice.

        :param algo: algorithm of choice between (1) hdbscan (2) aff_prop (3) k_means
        :param save_results: saves the clustering results inside the instance *results* dictionary
        :param kwargs: additional function keyword arguments
        :return: a tuple of (labeled_queries DataFrame, the clustering object)
        """
        doc2vec_model_params = config['doc2vec']['def_params']
        model_dir = config['global']['saved_models_dir']
        doc2vec_model_filename = model_dir + \
                                 self.csv_file_name + \
                                 self.current_df_date + \
                                 repr(doc2vec_model_params) + \
                                 config['doc2vec']['model_filename_postfix']

        if os.path.isfile(doc2vec_model_filename) and config['global']['always_load_files']:
            self.load_doc2vec_model(doc2vec_model_filename)
        else:
            self.fit_doc2vec()

        start = time.time()

        if algo == 'hdbscan':
            print('Clustering using HDBScan started')
            labeled_queries, model = self.hdbscan_clustering(**kwargs)
        elif algo == 'aff_prop':
            print('Clustering using Affinity Propagation started')
            labeled_queries, model = self.aff_prop_clustering(**kwargs)
        elif algo == 'k_means':
            print('Clustering using K-Means started')
            labeled_queries, model = self.k_means_clustering(**kwargs)
        else:
            raise Exception("Please specify either 'hdbscan', 'aff_prop' or 'k_means' for alg")

        running_time = (time.time() - start)
        minutes, seconds = divmod(running_time, 60)

        print(f'Clustering running time: {int(minutes)} minutes and {seconds:.2f} seconds')

        # result_name has 3 elements: (1) clustering algorithm used (2) kwargs used, if any (3) timestamp
        if save_results:
            now = datetime.datetime.now().strftime('%H%M%S')
            result_name = algo + str(kwargs) + now
            self.results[result_name] = {'labeled_queries': labeled_queries,
                                         'model': model}
        # TODO: check this:
        # if dump_to_csv:
        #     dump_to_csv()

        return labeled_queries, model

    def make_daily_clusters(self, score=True, **kwargs):
        if self.daily_dfs is None:
            self.filter_daily(**kwargs)

        print(f'\nClustering for {len(self.daily_dfs)} days')

        if score:
            daily_scores = OrderedDict()

        for day, df in self.daily_dfs.items():
            print(f'\n--- Performing clustering for {day} --->')
            sqc = SearchQueriesClustering(df)
            sqc.current_df_date = day
            sqc.aggregate_df()
            sqc.clean_search_queries()
            sqc.remove_i_will()
            sqc.make_clusters()
            self.daily_clusters[day] = sqc
            if score:
                scored_clusters = sqc.score_all_clusters()
                daily_scores[day] = scored_clusters
            sqc.document_process_details()
        if score:
            self.daily_scores = dict(daily_scores)

    #########################################
    ## Cluster Representation and Analysis ##
    #########################################

    def most_common_words_in_cluster(self, cluster_number,
                                     labeled_queries=None,
                                     n_common=5,
                                     display_relevant_queries='top',
                                     verbosity=25,
                                     show_non_relevant_examples=True):
        """
        Returns the n most common words in a labeled_queries dataframe
        display relevant queries would show queries of which common words appears.
        'top' would only return queries with only the top common word, 'all' would return queries with all common words.

        :param labeled_queries: Pandas DataFrame that is produced by a clustering function
        :param cluster_number: cluster number to show
        :param n_common: shows which n words are the omst common
        :param display_relevant_queries: show the queries that contain the first top common word
        :param verbosity: the number of search queries to show if `display_relebant_queries=True`
        :param show_non_relevant_examples: displays some exapmples without the first most common word
        :return: n most common
        """

        if labeled_queries is None:
            last_result = list(self.results.keys())[-1]
            labeled_queries = self.results[last_result]['labeled_queries']

        results = labeled_queries[labeled_queries['cluster_label'] == cluster_number]
        print(f'Number of queries in cluster #{cluster_number}: {len(results)} \n')
        all_words = results['query'].str.cat(sep=' ')

        c = Counter(all_words.split(' '))
        most_common = c.most_common(n_common)

        print('Most common words in all the queries:')
        pprint(most_common)

        # query display
        relevant_queries = []
        non_relevant = 0
        non_relevant_queries = []
        if display_relevant_queries == 'top':
            top_common = most_common[0][0]
            for row in results.iterrows():
                if top_common in row[1]['query']:
                    relevant_queries.append(row[1]['query'])
                else:
                    non_relevant += 1
                    non_relevant_queries.append(row[1]['query'])

            if verbosity is not None:
                num = verbosity
            else:
                num = ''

            c2 = Counter(relevant_queries)
            if verbosity is not None:
                c2 = c2.most_common(verbosity)
                print(f'\n{num} queries that contain the single most common word, "{top_common}":')
                pprint(c2)
            else:
                print(f'All queries that contain the single most common word, "{top_common}":')
                pprint(c2)

            print(
                f'\nThere were {non_relevant} queries in the cluster (out of {len(results)}) that did not contain the top common word.')
            if show_non_relevant_examples and non_relevant > 0:
                print(f'Random examples of such queries are:')
                pprint(np.random.choice(non_relevant_queries, size=10, replace=False).tolist())

        elif display_relevant_queries == 'all':
            print("'all' switch not yet implemented! :-]")

        elif display_relevant_queries is not None:
            raise Exception("Error: display_relevant_queries can either be set to 'all', or 'top'")

        return most_common

    def get_cluster(self, cluster_number, labeled_queries=None):
        """
        Gets a cluster from a labeled_queries table.

        :param cluster_number: cluster number to show
        :param labeled_queries: labeled_queries table to get the cluster from (if not specified, gets last result)
        :return: Cluster as a Pandas Series
        """
        if labeled_queries is None:
            last_result = list(self.results.keys())[-1]
            labeled_queries = self.results[last_result]['labeled_queries']

        df = self.df
        queries_col = config['queries_col']
        ns_col = config['num_search_col']

        if cluster_number not in labeled_queries['cluster_label'].values:
            print(f'Cluster label {cluster_number} does not exist (max={labeled_queries["cluster_label"].max()}).')
            raise Exception('No such cluster number ', cluster_number)
        else:
            cluster = labeled_queries[labeled_queries['cluster_label'] == cluster_number]

            def return_num_searches(row):
                query = row['query']
                num_searches = (df[df[queries_col] == query][ns_col]).sum()
                return num_searches

            cluster['num_searches'] = cluster.apply(return_num_searches, axis=1)

            return cluster

    def show_cluster(self, **kwargs):
        """
        Shows a cluster from a labeled_queries table.

        :param cluster_number: cluster number to show
        :param labeled_queries: labeled_queries table to show from (if not specified, shows from last result)
        :return: Cluster as a Pandas Series format
        """

        cluster = self.get_cluster(**kwargs)
        cluster_uniques = cluster.groupby('query').sum().to_dict()['num_searches']
        cluster_uniques = dict(sorted(cluster_uniques.items(), key=operator.itemgetter(1), reverse=True))
        name, _ = self.infer_cluster_name(cluster)
        name = '_'.join(name)

        print(f'Cluster number:\t{cluster.iloc[0]["cluster_label"]}')
        print(f'Cluster name:\t{name}')
        print('Unique queries:\t', end='')
        pprint(cluster_uniques)
        print('--------------------------------------------------')

    def show_some_clusters(self, labeled_queries=None):
        """
        Shows some clusters from a labeled_queries table.

        :param n_random_clusters: Show this amount oif clusters. if not set, showing until user quits.
        :param labeled_queries: labeled_queries table to show from (if not specified, shows from last result)
        :return:
        """
        if labeled_queries is None:
            last_result = list(self.results.keys())[-1]
            labeled_queries = self.results[last_result]['labeled_queries']

        print('\n\n\t --- Showing random clusters ---')
        already_shown = set()

        while len(already_shown) < labeled_queries['cluster_label'].nunique():
            print()
            random_cluster = np.random.choice(list(set(labeled_queries['cluster_label']) - already_shown))
            already_shown.add(random_cluster)
            self.show_cluster(cluster_number=random_cluster, labeled_queries=labeled_queries)
            more_to_show = labeled_queries['cluster_label'].nunique() - len(already_shown)
            if more_to_show == 0:
                break
            if input(f'{more_to_show} More clusters to show. Press Enter to continue, q to quit...').lower() == 'q':
                break

    def get_cluster_as_dict(self, cluster_number, include_similarities=False, *args, **kwargs):
        """
        Gets a cluster from a labeled_queries table in a form of a dictionary.
        Also inferring the cluster's name.

        :param cluster_number: cluster number to show
        :param include_similarities: include in the resulting dictionary field of similar names found in search corpus
        :return: Cluster as a dictionary
        """
        try:
            cluster = self.get_cluster(cluster_number, *args, **kwargs)
        except Exception:
            return None

        grouped = cluster.groupby('query').sum()
        # grouped.rename(columns={'num_searches': 'queries'}, inplace=True)
        d = grouped.to_dict()

        cluster_names, name_similarities = self.infer_cluster_name(cluster)
        cluster_name = '_'.join(cluster_names)
        d['cluster_name'] = cluster_name
        d['cluster_number'] = cluster.iloc[0]['cluster_label']

        d.pop('cluster_label')
        d['cluster_number'] = d.pop('cluster_number')
        d['cluster_name'] = d.pop('cluster_name')
        d['queries'] = dict(sorted(d.pop('num_searches').items(), key=operator.itemgetter(1), reverse=True))
        if include_similarities:
            d['name_similarities'] = name_similarities

        return d

    def infer_cluster_name(self,
                           cluster,
                           method=config['infer_name']['method'],
                           max_n_names=config['infer_name']['max'],
                           n_similarities=config['infer_name']['n_sim']):
        """
        Tries to infer the name(s) of the cluster according to (1) most common word (2) similarity to most common word.
        While the most common word(s) are the obvious choice, similarities may be inspected for further insight.

        :param cluster: a single cluster, carved out of labeled_queries
        :param method: naming calculation method. (1) 'half' (2) 'magnitude' (3) 'lda'
        :param max_n_names: max number of names to extract
        :param n_similarities: number of name similarities (per name) to include in the output
        :return: (2 * n_names) names, according to (1) most common words and (2) their similarities
        """
        df = self.df
        qu_col = self.queries_col
        ns_col = config['num_search_col']

        # actual words
        all_words = []
        for query in cluster['query']:
            num_searches = df[df[qu_col] == query][ns_col]
            for word in set(query.split(' ')):
                for _ in range(num_searches.sum()):
                    all_words.append(word)
        c = Counter(all_words)
        commons = c.most_common(max_n_names)

        if config['global']['verbose']:
            print(f'Inferring cluster name out of top {max_n_names} which are:')
            pprint(commons)

        selected = [commons[0][0]]
        if len(commons) > 1:
            for i in range(1, max_n_names):

                if method == 'magnitude':
                    if commons[i - 1][1] % commons[i][1] < 10:
                        selected.append((commons[i][0]))
                    else:
                        break

                elif method == 'half':
                    if commons[i - 1][1] / commons[i][1] < 2:
                        selected.append((commons[i][0]))
                    else:
                        break

                # TODO: look for a way to have n_top_words calculated automatically according to some score
                elif method == 'lda':
                    lda.lda_top_words(df[qu_col], n_top_words=2)

                else:
                    raise Exception("'method' should be set to either (1) 'magnitude', 'half' or 'lda'!")

        if config['global']['verbose']:
            print(f'Inferred name: {"_".join(selected)}')

        # similar words
        similars = []
        for word in selected:
            try:
                sims = self.doc2vec_model.wv.most_similar(word)
                sims = sims[:n_similarities]
                similars.append(sims)
            except Exception as e:
                if config['global']['verbose']:
                    print(f'Warning: {e}')

        return selected, similars

    def name_all_clusters(self, labeled_queries=None, include_similarities=True, return_pandas=False, dump_csv=True):
        """
        Name all clusters in a given labeled_queries table

        :param labeled_queries: a labeled_queries Pandas Series (if not provided, automatically fetches the last result)
        :param include_similarities: also include name similarities from from word2vec corpus
        :param return_pandas: return a pandas DataFrame (instead of a dict)
        :param dump_csv: dumps results to a file. Filename is composed of (1) original csv file name (2) model name and
        properties (3) _named_clusters_.csv postfix.
        :return: a list of all named clusters, each of which is a dictionary
        """

        if labeled_queries is None:
            last_result = list(self.results.keys())[-1]
            labeled_queries = self.results[last_result]['labeled_queries']

        min_clusters = labeled_queries['cluster_label'].min()
        max_clusters = labeled_queries['cluster_label'].max()

        print('Naming all clusters', end=' ... ')

        named_clusters = {}
        for n in range(min_clusters, max_clusters + 1):
            try:
                dict_cluster = self.get_cluster_as_dict(n, include_similarities=include_similarities)

                cluster_name = dict_cluster['cluster_name']
                if cluster_name in named_clusters.keys():
                    new_queries = Counter(named_clusters[cluster_name]['queries']) + Counter(dict_cluster['queries'])
                    named_clusters[cluster_name]['queries'] = dict(new_queries)
                else:
                    named_clusters[cluster_name] = dict_cluster
            except Exception as e:
                print(f'No such cluster number {n}')

        named_clusters_df = pd.DataFrame(named_clusters).T

        if dump_csv:
            last_result_model = repr(self.results[last_result]['model']).replace('\n', '').replace(' ', '')
            filename = self.csv_file_name + '_' + last_result_model + '_named_clusters_.csv'
            if len(filename) > 255:
                extra = len(filename) - 255
                last_result_model = last_result_model[:-extra]
                filename = self.csv_file_name + '_' + last_result_model + '_named_clusters_.csv'
            self.dump_csv(named_clusters_df, filename)

        if return_pandas:
            return named_clusters_df.T
        else:
            return named_clusters

    def score_cluster(self, cluster_number, method=config['scoring']['method']):
        """
        Scores a cluster 'quality' according to some method
        :param cluster_number: cluster number (as identified in a labeled_queries DataFrame)
        :param method: (1) 'query_proportion': calculates the proportion of the top repeating query out of all queries
        (2) 'word_proportion': calculates the proportion of words in queries
        :return:
        """
        cluster = self.get_cluster(cluster_number)
        queries_col = config['queries_col']
        ns_col = config['num_search_col']

        if method == 'query_proportion':
            total_searches = cluster[ns_col].sum()
            max_searches = cluster[ns_col].max()
            top_proportion_query = max_searches / total_searches
            return top_proportion_query

        elif method == 'word_proportion':
            all_words = Counter()
            for row in cluster.itertuples():
                multiplier = int(row.num_searches)
                c = Counter(row.query.split(' '))
                row_counts = Counter()
                for k in c.keys():
                    row_counts[k] = c[k] * multiplier
                all_words += row_counts
            most = sum(list(zip(*all_words.most_common(2)))[1])
            total = sum(all_words.values())
            return most / total



        else:
            raise Exception('Scoring method should be either set to (1) query_proportion or (2) word_proportion')

    def score_all_clusters(self, labeled_queries=None, dump_csv=False, return_pandas=False):
        """
        Scores all clusters.

        :param labeled_queries: a labeled_queries Pandas Series (if not provided, automatically fetches the last result)
        :param dump_csv: dumps results to a file. Filename is composed of (1) original csv file name (2) model name and
        properties (3) _named_clusters_.csv postfix.
        :param return_pandas: return a pandas DataFrame (instead of a dict)
        :return: a dictionary of all clusters and their score
        """
        if labeled_queries is None:
            last_result = list(self.results.keys())[-1]
            labeled_queries = self.results[last_result]['labeled_queries']
        print('Scoring all cluster for this date', end=' ... ')
        named_clusters = self.name_all_clusters(labeled_queries=labeled_queries, return_pandas=True, dump_csv=False)
        scored_clusters = named_clusters.copy().T
        scored_clusters['score'] = scored_clusters['cluster_number'].apply(self.score_cluster)
        scored_clusters['score'] = scored_clusters['score'].apply(lambda x: np.round(x, 3))

        scored_clusters_df = pd.DataFrame(scored_clusters)

        if dump_csv:
            last_result_model = repr(self.results[last_result]['model']).replace('\n', '').replace(' ', '')
            filename = self.csv_file_name + '_' + last_result_model + '_scored_clusters_.csv'
            if len(filename) > 255:
                extra = len(filename) - 255
                last_result_model = last_result_model[:-extra]
                filename = self.csv_file_name + '_' + last_result_model + '_scored_clusters_.csv'
            self.dump_csv(scored_clusters_df, filename)

        print('done')

        if return_pandas:
            return scored_clusters_df
        else:
            return scored_clusters

    def create_daily_scores_table(self):
        """
        Crates a scores tables for all days and saves the result in self.daily_scores_table.
        """
        if self.daily_scores is None:
            raise Exception('You need to first make daily clusters.'
                            'Use make_daily_clusters with score=True.')

        # for storing all cluster names found during daily scoring to be used as dataframe index
        all_cluster_names = set()

        daily_scores_dict = {}
        for day, d in self.daily_scores.items():
            daily_scores_dict[day] = [d['cluster_name'], d['score']]
            for cluster_name in d['cluster_name'].tolist():
                all_cluster_names.add(cluster_name)

        self.all_cluster_names = all_cluster_names

        daily_scores_table = pd.DataFrame(all_cluster_names)
        daily_scores_table.set_index(0, inplace=True)
        for date, names_scores_list in daily_scores_dict.items():
            for name_score in zip(*names_scores_list):
                cluster_name = name_score[0]
                score = np.round(name_score[1], 3)
                daily_scores_table.loc[cluster_name, date] = score
        daily_scores_table.fillna(value=0, inplace=True)

        self.daily_scores_table = daily_scores_table
        return daily_scores_table

    def get_cluster_daily_scores(self, cluster_name: str):
        """
        Gets scores for a cluster based on its name representation, if one such exists.
        :param cluster_name:
        :return:
        """
        if self.daily_scores_table is None:
            raise Exception('You need to first create a daily scores table.')

        return self.daily_scores_table[self.daily_scores_table.index == cluster_name]

    def get_cluster_daily_queries(self, cluster_name: str):
        if self.daily_scores is None:
            raise Exception('You need to first make daily clusters.')

        daily_queries = OrderedDict()
        for day, item in self.daily_scores.items():
            daily_queries[day] = item[item['cluster_name'] == cluster_name]['queries']

        daily_queries = pd.DataFrame(daily_queries)
        daily_queries = dict(daily_queries)
        return daily_queries

    # def get_cluster_daily_queries_and_scores(self, cluster_name):
    #     queries = self.get_cluster_daily_queries(cluster_name)
    #     scores = self.get_cluster_daily_scores(cluster_name)

    ###########
    ## Utils ##
    ###########

    @staticmethod
    def dump_csv(df, filename):
        """
        Helper function to dump Pandas DataFrames to csv files

        :param df: Pandas DataFrame
        :param filename: file name to dump to
        """
        with open(filename, 'w') as f:
            df.to_csv(f)

    def filter_daily(self,
                     df=None,
                     date_column=config['date_col'],
                     interval=config['daily']['interval'],
                     max_total_days=config['daily']['max_total_days'],
                     skip_start=config['daily']['skip_start']):
        """
        Splits a multi-day DataFrame or csv into daily DataFrames.

        :param df: is the DataFrame to be split. Should come from df in the instance.
        :param date_column: column of the date
        :param interval:  the gap between days to take out of a data frame in a daily process
        :param max_total_days: the maximum number of days (hence daily data frames)
        :param skip_start: first dates in the data to skip
        """
        if df is None:
            df = self.df

        print('Splitting data into days', end=' ... ')
        unique_days = sorted(set(df[date_column]))
        daily_dfs = OrderedDict()
        for i, day in enumerate(unique_days):
            if i < skip_start - 1:
                continue

            if len(daily_dfs) >= max_total_days:
                break

            if (i - skip_start) % interval == 0:
                daily_df = df[df[date_column] == day]
                daily_dfs[day] = daily_df

        self.daily_dfs = dict(daily_dfs)
        print('done')

    @staticmethod
    def filter_one_day(sqc_obj):
        """
        Returns one random day from a multi-day DataFrame
        :param sqc_obj: SearchQueryClustering object
        :return: SearchQueryClustering object
        """
        print('Selecting one day from the DataFrame', end=' ...')
        first_date = list(sqc_obj.filter_daily().keys())[0]
        sqc_obj.filter_daily()[first_date]
        print('done')
        return sqc_obj

    def document_process_details(self, filename=None):
        """
        Documenting the details of the process to a text file.

        :param filename: filename to save to. Defauls is current timestamp.
        :return: Logging of filename the details were saved to.
        """
        if filename is None:
            model_dir = config['global']['saved_models_dir']
            filename = model_dir + str(time.time()) + '__process_details.txt'

        with open(filename, 'w') as f:
            f.write(f'csv file: {self.csv_file_name}')
            f.write('\n')
            f.write(f'doc2vec model: {self.doc2vec_model}')
            f.write('\n')
            f.write(f'clustering model:  {self.clustering_model}')

        return f'Saved process details to {filename}'

    def get_available_cluster_names(self):
        return self.all_cluster_names

    ##############
    ## Plotting ##
    ##############

    def plot_stats(self, **kwargs):
        """
        Plots graphic information about individual words or complete queries in the searches corpus.

        :param top_n: top n words to show
        :param separate: if True, returns most common *words*. Otherwise returns most common *complete queries*.
        """
        eda_utils.plot_stats(self.df, **kwargs)

    def plot_daily_scores(self, cluster_names: list):

        clusters = []
        for name in cluster_names:
            clusters.append(self.get_cluster_daily_scores(name).T)

        clusters = pd.concat(clusters, axis=1)
        sns.lineplot(data=clusters, dashes=False)
        plt.title(
            f"Scores with {len(cluster_names)} clusters for {len(clusters)} days. lim_df={config['global']['limit_dataframe']}")
        plt.show()

    def plot_n_random_scores(self, n=5):
        random_cluster_names = np.random.choice(list(self.get_available_cluster_names()), n)
        self.plot_daily_scores(random_cluster_names)
