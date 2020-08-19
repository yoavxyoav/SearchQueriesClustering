# Search Queries Clustering 



By Yoav Vollansky (ver. 0.1)

[TOC]

### 0. Introduction

SearchQueriesClustering is a class that performs clustering of search queries. The class supports clusteing of tabular data of which one column holds the search queries. A Gensim's Doc2Vec representation of that column is undergoing unsupervised clustering, and each cluster gets a representation (in a form of an inferred textual name). Score is then calculated for each cluster, based on its consistency of queries which could stand for cluster "purity". A typical further analysis would be to employ such process over some span of time and comparing clusters scored over days, weeks or moths.

### 1. Installation 

* Using Conda or Pip, use requirements.txt in a new environment to install all required packages. 
* Copy the csv files you intend to work with to the `data` directory (*default: ./data/*)

### 2. Dataset

The class expect tabular data with the following columns: `search_date`, `search_query`, `number_of_searches`, `searches_with_clicks`, `total_number_of_search_results`, `listings_with_orders`, `orders`, `most_common_sub_category_name`,  and `most_common_category_name`. For this version, some of the columns are aggregated within the script (in `SearchQueriesClustering.aggregate`) while others are filtered out.

An example of a Google BigQuery query that would produce a desirable output to be the input to the class:

```sql
select search_date, search_query, 
sum(number_of_searches) number_of_searches,
sum(searches_with_clicks) searches_with_clicks,
avg(total_number_of_search_results) total_number_of_search_results,
sum(listings_with_orders) listings_with_orders,
sum(orders) orders,
sum(revenue),
most_common_sub_category_name, 
most_common_category_name
from `itc-data-science.dwh.search_keywords`
where search_query is not null
and search_date >'2020-01-01'
group by search_date, search_query, most_common_sub_category_name,most_common_category_name
order by 2
```

***Note***: *change `search_date > ... ` according to requirements of time span.*

### 3. General Clustering vs. Daily Clustering

There are two ways to work with the class:

1. **General query clustering**: working on a single data frame, clustering it as it is, whether it has multiple days or not. This is done by using `make_clusters()`.
2. **Daily query clustering**: working on a data frame but splitting it by date into many smaller data frame in the process. This happens automatically in the pipeline when using `make_daily_clusters()`.

[Read below how to use both ways](#6.-usage-and-workflow ). 

Clustering itself is done using either K-Means, Affinity Propagation or HDBscan. At this point the safest way to work is by using HDBScan as it was the method tested the most (especially with *daily clustering*).

While when working with general query clustering and `make_clusters()` only give us one high-level SerchQueriesClustering instance that only contains a single model and DataFrame to deal with, when working with daily queries clustering and `make_daily_clusters()`', our instance will contain more attributes that have per-day information. One of those attributes, for example, is `daily_clusters`, <u>which is a dictionary of SearchQueriesClustering instances, **one for each day**</u>. 

The core idea is that when working with *daily clustering*, we are in essence carrying a **separate** process for each daily segment of the data. For each day, the process will perform <u>cleaning, aggregating, doc2vec embedding, clustering, naming and scoring.</u> 

*Note: in case of daily clustering, pre precessing and cleaning was placed in the pipeline <u>after</u> splitting to days, since such splitting requiers the date column. If date column does not exist when splitting, an error would occur. If you intend to manually split to days, make sure to follow this practice.*

### 4. Configuration

Most of the configuration is contriled in the `config.yaml` file. The preferences are as follows:

In the root of the config, you can define some most useful columns as they appear in the imported csv file: `queries_col` is the queries column, `date_col` is the date and `num_search_col` is the number of searches column. 

*  `global` preferences
  * `always_load_files` will load pre-saved doc2vec models and clustering pickle files if they exist in the `data_dir` directory and fit to the current parameters, as defined in `doc2vec: def_params` or clustering specific parameters in `k_means: def_params`, `aff_prop: def_params` and `hdbscan: def_params`. This is helpful for fast pipeline testing and rerunning of pre-saved models.
  * `always_save_files` will overwrite existing files with similar parameters as mentioned above. 
  * `drastic_cleanup` will delete all models and pickles. (This is not anything which is automated. It can be used be used as a conditional in `main()`, for example, in conjunction to the class's static method `drastic_cleanup`.
  * `verbose` will cause to emit extra output when running some functions.
  *  `always_plot` will plot some initial EDA visualisations when after loading a csv file (*Note: only use for debugging, definitely disable in automated use cases as it requires user interaction*)
  * `data_dir` is where input csv files reside
  * `saved_models_dir` is where dov2vec .model files and clustering objects .pickle files are saved to
  * `limit_datafreme` is the sample size of the input data frame. Best used only for development and testing, as models do require large, complete data, to work optimally. Set to `None` to leave the data frame as is.
  * `dataframes_prefs` will set Pandas output according to the next two properties, `max_rows` and `max_cols`.

*  `agg` settings

  During preprocessing, `aggregate()` should be used , with columns set to `sum` being accumulated, with `mean` returning their average, and with `first` silly returning the first value in the column, all with respects to each specific **date**. 

* TF-IDF related preferences are specified in `tf_idf` (initial development, suspended at the moment)

* `doc2vec` settings

  * `def_params` for the model. [Read here](https://radimrehurek.com/gensim/models/doc2vec.html) for details. Other parameters can be passed as `**kwargs` to the training function (or should be hard coded and set in `config.yaml` to it if used often).
  * `mode_filename_postfix` to be added at the end of the saved model filename 
  * `tagged_documents_features` are other features in the input csv that will be used to tag the documents. [Read more about TaggedDocument here](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.TaggedDocument). 

* `k_means`, `aff_prop` and `hdbscan` are all clustering algorithm specific parameters. 

* `infer_name` are preference that related to the way a cluster is represented, i.e which name it gets after clustering is performed:

  * `max` is the maximum number of words in the title of a cluster
  * `n_sim` is the number of similarities to each of the names inferred, based on word2vec similarity.
  * `method` can be one of the following: `magnitude`, `half` or `lda`. See `infer_cluster_name()` below for more details.

* `daily` controls preferences for `filter_daily()` as follows:

  *  `skip_start` skips first dates in the data (so if our first day in the data is `2020-01-01` and we would like to start processing from `2020-01-04`, we would set the value to `3`)
  * `max_total_days` is the number of days we would want in our daily process. It is possible to get a smaller number of days if the dataset spans less days than we set for.
  * `interval` indicates the gap between days we take out of a data frame in a daily process. If it is set to `7` (and `skip_start=0`), then our first day would be `2020-01-01`, second `2020-01-08`, third `2020-01-15` etc. 

* `scoring` controls the settings for `score_cluster()`. It can be set to either `query_proportion` or `word_proportion`. More details below. 

### 5. Getting familiar with the SearchQueriesClustering class

Most of the functionality of the class is built into a pipeline and is executed automatically via wrapped functions. The individual functions can also be used manually by being called from the main module if pipeline customisation is needed (although currently this is still prone to breakage, so make sure to debug the process).

The specification of the class was thoroughly documented in this section both for the sake of any future development efforts, as well all to users of the class that that wish to customise the pipeline. 

#### 5.1 Class Attributes

* `csv_file_name` is the name of the acv file that the data is loaded from. Mainly used for unique file naming when saving trained doc2vec models and clustering object pickle files to drive. It is set to `__session_dataframe__` if the data was loaded from memory and not from a csv file.
*  `data` is the argument as passed to the class constructor, and it can either be a pd.DataFrame or a file name.
* `current_date_df` is the date of the current data that is associated with the class instance if working in *daily clustering* mode (i.e if an original DataFrame was split into days)
* `df` is the DataFrame in the current instance (and it can be either daily or general). It uses the `_loader()` function to determine whether the argument that was passed to the constructor, and no resides in `data`, is a filename or a DataFrame in memory, and loads the data accordingly.
* `queries_col` is the default name for the queries column in `df`, and can be changed in `config.yaml`
* `doc2vec_model` is the trained and fit model, saved in the instance level for convenient checking of parameters etc. Helpful for avoiding the need to retrain a model in case we test a few clustering approaches in the same session. 
* `similarity_arr` is the similarity array resulted from the TF-IDF process (not yet thoroughly supported in this version)
* `doc2vec_arr` is the raw matrix produced bu the doc2vec model. Created along with the `doc2vec` attribute upon training and fitting.
* `unique_search_queries` holds the unique search queries for an instance `df`. 
* `tokenized_queries` holds the tokenized queries of an instance `df` queries column. 
* `clustering_model` holds the model that was used to cluster a `doc2vec_arr`. Useful for quick lookup of parameters. 
* `results` holds all the results for a given DataFrame, if in a session many clustering approaches were tested for comparison. For example, if for some DataFrame we tried HDBscan with some set of parameters, then changed those parameters and clustered again, and then changed to Affinity Propagation which is a different algorithm altogether, all the results are saved to this dictionary for quick and convenient inspection and comparison. 
* `daily_dfs` holds all the separated DataFrames per day once split by `filter_daily()`.
* `daily_clusters`, similarly to `daily_dfs`, is a dictionary of the resulted **<u>per-day</u>**  SearchQueriesClustering instances resulted by `make_daily_clusters()`. 
* `daily_scores` is a dictionary for scores as calculated by `score_all_clusters()`.
* `daily_scores_table` is a tabular version of `daily_scores`
* `all_cluster_names` remembers all the names of all clusters, for all days, which is essential for building the `daily_scores_table` in `create_daily_scores_table()`.

#### 5.2 Class Methods

##### 5.2.1 Initialisation, data loading and preprocessing

```python
def _loader(self)
```

is an internal method that adds flexibility to the data loading. Since we can give the class constructor either a string indicating a csv filename or an actual DataFrame from memory, `_loader` is determining which of the two we passed and uses the appropriate way to load the data.

```python
def load_data(csv_file, remove_na=True)
```

simply loads the data ins case `_loader` has datelined we are loading a csv file. 

```python
def drastic_cleanup(remove_models=True, remove_pickles=True)
```

removes pre-saves files from working directory. It is not part of the internal pipeline at the moment, but can be conveniently used as a conditional in the main module.

```python
def remove_na(df)
```

is removing NaN rows from the DataFrame. It is performed the earliest in the preprocessing pipeline and is usually called by `load_data()`.

```python
def remove_i_will(self)
```

will remove any query that starts with "I will", as we would like to try and filter out non-buyer related queries. 

```python
def clean_search_query(query)
```

cleans individual queries by removing non-alpha numerics, stripping white space from sides and lowercasing.

```python
def clean_search_queries(self, clean=True, remove_numeric=True)
```

is a wrapper function around `clean_search_query` that runs over all queries in the DataFrame and also drops any purely numeric queries if `remove_numeric=True`.

##### 5.2.2 TF-IDF

*Functions in this section are currently not utilised in the pipeline and were not thoroughly tested for this version.* 

```python
def create_tfidf_similarity(self, queries_series, max_samples=config['tf_idf']['max_samples']):
```

creates a similarity matrix based on TF-IDF of a corpus.

```python
def find_unique_queries(self):
```

gets all uniques from a DataFrame.

```python
def show_n_similar_queries(self, query, n_similar):
```

once a tf-idf similarity array is created, this method returns most *n* similar queries to `query`.

```python
def find_all_non_zero_score_similar_queries(self, query)
```

same as above, only instead of showing *n* similarities, shows all non-zero-score similarities.

##### 5.2.3 Doc2Vec

Many of the methods in this section (as well as some of the clustering related methods) check for dependencies in terms of required instance attributes, and automatically call other methods in case these attributes are not populated with a value (usually they are set to `None` upon instantiation). This enables a one-shot call for wrapper methods, such as `make_daily_clusters()`, without having to worry about running methods that are required to be executed priorly to other methods.

```python
def tokenize_queries(self, pre_process=True)
```

tokenises the search queries column in the in the instance DataFrame and saves it as an attribute.

```python
def create_tagged_documents(self,
                            tag_sequential=True,
                            tagging_features=config['doc2vec']['tagged_documents_features'])
```

creates a TaggedDocument object which is required for Gensim's Doc2Vec fitting. `tag_sequential=True` is the straightforward way to go when doing the tagging, and all it essentially does is tagging each document with an increasing index (0, 1, 2, etc.). A more experimental approach would be to tag the documents using other metadata features from the dataset, such as number of searches or sub category. These other features can either be passed to `tagging_features` manually, or set in `config.yaml` when running the entire pipeline as a whole using `make_clusters()` or `make_daily_clusters(`).

See [TaggedDocument](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.TaggedDocument) for more details. 

```python
def fit_doc2vec(self,
                filename=None,
                model_params=config['doc2vec']['def_params'])
```

is fitting the doc2vec model with default parameters as specified in `config.yaml`. filename can explicitly specified, but if not, it is inferred from the scv filename, date of queries (if *daily clustering* is performed) and model parameters, in order to create a unique filename to save the model to.

```python
def load_doc2vec_model(self, filename=None):
```

loads a pre-saved doc2vec model from the specified file. If filename=None, the file name is inferred from the csv file name, date of queries (if *daily clustering* is performed) and model parameters. Note that  `config.yaml` `always_load_files` setting should be set to `True` in order for files to load (this condition is checked in `make_clusters()`).

```python
def marry_queries_to_labels(queries, cluster_labels)
```

is a helper function that takes the queries column and concatenates them to the cluster labels  (numeric ones) resulted from clustering. It outputs a Pandas DataFrame.

```python
def assert_doc2vec_model_exists(self)
```

is a helper function that checks whether the instance attributes `doc2vec_model` and `doc2vec_arr` have both been populated. This is used down the pipeline within each clustering method, since the clustering process is based on having doc2vec available. 

##### 5.2.4 Clustering

The class currently supports three clustering algorithms: K-Means, Affinity Propagation and HDBScan. Future versions may test more approaches.

```python
def k_means_clustering(self,
                       k=config['k_means']['def_params']['k'],
                       load_file=True,
                       **kwargs)
```

```python
def aff_prop_clustering(self,
                        damping=config['aff_prop']['def_params']['damping'],
                        affinity=config['aff_prop']['def_params']['affinity'],
                        load_file=True,
                        **kwargs)
```

```python
def hdbscan_clustering(self,
                       cluster_selection_epsilon=config['hdbscan']['def_params']['cluster_selection_epsilon'],
                       min_samples=config['hdbscan']['def_params']['min_samples'],
                       min_cluster_size=config['hdbscan']['def_params']['min_cluster_size'],
                       allow_single_cluster=config['hdbscan']['def_params']['allow_single_cluster'],
                       cluster_selection_method=config['hdbscan']['def_params']['cluster_selection_method'],
                       leaf_size=config['hdbscan']['def_params']['leaf_size'],
                       load_file=config['global']['always_load_files'],
                       **kwargs)
```

At the moment, parameters should be set in `config.yaml`. Ones that are not set in the config file can be passed in a manual process via `**kwargs`, and if found particularly important should be hard coded to the above methods and set in the config.

**Note** that each of the clustering methods above is also checking for the existence of priorly saved clustering result (as a pickle file) and loading it if `load_file=True`. 

```python
def make_clusters(self,
                  algo='hdbscan',
                  save_results=config['global']['always_save_files'],
                  **kwargs)
```

is a wrapper function around the clustering algorithms, having hdbscan a the default choice as it was tested the most in this version. If `save_results=True`, clustering object files will be saved *and overwrite* priorly saved pickle files for data from the same csv file and  with the same clustering algorithm-specific preferences (since these factors, when put together in a string, is how the filename is formed).

```python
def make_daily_clusters(self, score=True, **kwargs)
```

is a wrapper method that runs the entire pipeline for *daily clustering*. 

First it loads the entire csv file into a higher-level SearchQueriesClustering  instance, and then splits the input data into days and crates SearchQueriesClustering instances with each of the daily DataFrames, which are stored inside the `daily_clusters` attribute dictionary. In turn, <u>for each day</u> it aggregates the data, cleans it, and uses `make_clusters()` on each of the daily instances. If `score=True`, it also scores the clusters and stores the results in the higher-level SearchQueriesClustering instance `daily_scores` attribute. 

##### 5.2.5 Cluster naming, scoring and analysis

**Note** that for analysis methods that are performed on a single DataFrame, execution **must be** **within** **a daily SearchQueriesClustering instance**, since if *daily clustering* was performed (as opposed to the higher-level instance, *general clustering* way), results can only be found within such instances. In other words, some analysis can only be performed per day, and if that is the case, it should be executed within one of the the `daily_clusters` dictionary SearchQueriesClustering instances. Executing the methods on a higher level SearchQueriesClustering will not work.

```python
def most_common_words_in_cluster(self, cluster_number,
                                 labeled_queries=None,
                                 n_common=5,
                                 display_relevant_queries='top',
                                 verbosity=25,
                                 show_non_relevant_examples=True)
```

displays the *n* most common individual words in a `labeled_queries` DataFrame, which is the result of any of the clustering algorithms. A cluster number should be passed to the method, where possible numbers are the labels in `labeled_queries`.

If a `labeled_queries` DataFrame is not specified, the last results from instance `results` dictionary will be shown. If `display_relevant_queries` is set (currently on the the 'top' switch is implemented), queries that contain the first most common word will be shown as well in output. 

```python
def get_cluster(self, cluster_number, labeled_queries=None)
```

is a helper function to get a cluster. If a `labeled_queries` DataFrame is not specified, the last results from instance `results` dictionary will be shown. `cluster_number` should be passed explicitly as a keyword argument. Returns a DataFrame.

```python
def show_cluster(self, **kwargs)
```

shows a cluster number, its name (calls `infer_cluster_name()`) and unique queries in the cluster.  If a `labeled_queries` DataFrame is not specified, the last results from instance `results` dictionary will be shown. Uses `get_cluster()` to fetch the cluster data.  

```python
def show_some_clusters(self, labeled_queries=None)
```

is a wrapper around `show_cluster()` that displays random clusters for analysis purposes. f a `labeled_queries` DataFrame is not specified, the last results from instance `results` dictionary will be shown.

```python
def get_cluster_as_dict(self, cluster_number, include_similarities=False, *args, **kwargs)
```

will return the cluster in a dictionary format, including word2vec similarities if `include_similarities=True`. Used in `name_all_clusters()`.

```python
def infer_cluster_name(self,
                       cluster,
                       method=config['infer_name']['method'],
                       max_n_names=config['infer_name']['max'],
                       n_similarities=config['infer_name']['n_sim'])
```

gets a cluster as produced from `get_cluster()` and names it according to a naming method (default set in `config.yaml`). 

`max_n_names` is the maximum number of words in the name (e.g *logo_design* is a 2 word name while just *dropshipping* is a 1 word name).

`method` can be: 

1. `magnitude`: counts the number of individual words (not complete search queries) in the cluster and selecting words that appear in the same order of magnitude. For example, if the topmost counted word appears 1500 times, the second appears 1200 and the third 300, then only the first two words will be selected since they are in thousands. 
2. `half`: keeps including words that are appear not less than half times than the word added before that. For example, if the topmost word appears 1200 times, and the second topmost appears only 500 (less than 1200/2=600), then it would not be part of the name. 
3. `lda`: according to an lda algorithm. Still tested in this version. 

```python
def name_all_clusters(self, labeled_queries=None, include_similarities=True, return_pandas=False, dump_csv=True):
```

is a wrapper function to names all clusters and returns the results either as a dictionary or a Pandas DataFrame. Note that the method calls `get_cluster_as_dict()`, which in turn calls `infer_cluster_name()` (I,e, it does not call the latter directly).

```python
def score_cluster(self, cluster_number, method=config['scoring']['method'])
```

will score a given cluster in terms of 'purity' or 'consistency'. A "stable cluster" would be such that has a relatively high score over time. 

`method` default value is set in `config.yaml` and can be one of the following:

1. `query_proportion` scores the cluster by calculating the rate of the topmost repeating query out of all queries.
2. `word_proportion` scores the cluster by calculating the rate of the topmost repeating word out of all words in all the queries in the cluster. 

```python
def score_all_clusters(self, labeled_queries=None, dump_csv=False, return_pandas=False)
```

will score all clusters in a `labeled_queries` DataFrame (or if `None`, then the DataFrame in the last instance`results` entry) and return either a dictionary or a DataFrame according to `return_pandas`. This function is used within the `make_daily_clusters()` pipeline, then saving the results to the instance attribute `daily_scores`.

```python
def create_daily_scores_table(self)
```

based on the instance attribute `daily_scores` (which is in the pipeline of `make_daily_clusters()`), crating a scores table to all days. Stores the result in the instance attribute `daily_scores_table`.

```python
def get_cluster_daily_scores(self, cluster_name: str)
```

Gets scores over days for a cluster based on its name representation, if one such exists.

```python
def get_cluster_daily_queries(self, cluster_name: str)
```

Gets queries over days for a cluster based on its name representation, if one such exists.

```python
def get_cluster_daily_queries_and_scores(self, cluster_name)
```

is a unification of the two functions above. Not implemented yet in this version.

##### 5.2.6 Helper functions

```python
def dump_csv(df, filename)
```

is for saving of clustering output to a csv file.

```python
def filter_daily(self,
                 df=None,
                 date_column=config['date_col'],
                 interval=config['daily']['interval'],
                 max_total_days=config['daily']['max_total_days'],
                 skip_start=config['daily']['skip_start'])
```

is filtering a multi-day data frame into separate days, saving the results to the instance attribute `daily_dfs`.

*  `skip_start` skips first dates in the data (so if our first day in the data is `2020-01-01` and we would like to start processing from `2020-01-04`, we would set the value to `3`)
*  `max_total_days` is the number of days we would want in our daily process. It is possible to get a smaller number of days if the dataset spans less days than we set for.
*  `interval` indicates the gap between days we take out of a data frame in a daily process. If it is set to `7` (and `skip_start=0`), then our first day would be `2020-01-01`, second `2020-01-08`, third `2020-01-15` etc. 

```python
def filter_one_day(sqc_obj)
```

is a helper function that returns one random day from a multi-day DataFrame.

```python
def document_process_details(self, filename=None)
```

documents (saves to a text file named according to parameters etc,  or simply to the string passed to`filename`) the details of the process or pipeline. Example of the file:

```yaml
csv file: some_data.csv
doc2vec model: Doc2Vec(dm/m,d75,n5,w2,mc2,s0.001,t3)
clustering model:  HDBSCAN(algorithm='best', allow_single_cluster=False, alpha=1.0,
        approx_min_span_tree=True, cluster_selection_epsilon=0,
        cluster_selection_method='leaf', core_dist_n_jobs=4,
        gen_min_span_tree=False, leaf_size=5000,
        match_reference_implementation=False, memory=Memory(location=None),
        metric='euclidean', min_cluster_size=2, min_samples=1, p=None,
        prediction_data=False)
```

This method is called from inside the `make_daily_clusters()` pipeline.

```python
def get_available_cluster_names(self)
```

would return all available clusters names in a *daily clustering*. (This value is currently set inside `create_daily_scores_table` but naturally should not. Needs to be improved in next version.) 

##### 5.2.7 Visualization

```python
def plot_stats(self, **kwargs)
```

is useful for EDA, right after loading the data.

```python
def plot_daily_scores(self, cluster_names: list)
```

plots a line plot for clusters scores in `cluster_names`. Make sure that clusters are named and scored.

```python
def plot_n_random_scores(self, n=5)
```

plots *n* random clusters scores based on `plot_daily_scores()`

### 6. Usage and Workflow 

There are two ways the class can be used: [general clustering and daily clustering (see above)](#3. General Clustering vs. Daily Clustering). Following is a demonstration of how to use both.

#### 6.1 General clustering

*General clustering* requires the user to manually create the pipeline, and would usually be best for a one-shot DataFrame clustering, whether that DataFrame consists of one day or more. This would be good for debugging or interactive mode data exploration. 

An example of *general clustering* a would look as follows:

```python
# cleaning directory on startup (use carefully)
if config['global']['drastic_cleanup']:
  SearchQueriesClustering.drastic_cleanup()

# crating an instance and loading the data
sqc = SearchQueriesClustering(DATA_FILE)

# aggregation 
sqc.aggregate_df()

# cleanup
sqc.clean_search_queries()
sqc.remove_i_will()

# crating a doc2vec embedding and performing HDBScan clustering
labeled_queries, clustering_obj = sqc.make_clusters()

# saving model details to file
sqc.document_process_details()

# showing some clusters
sqc.show_some_clusters()

# fetching a cluster as a dict
sqc.get_cluster_as_dict(5)

# manual inference of cluster name
some_cluster = sqc.get_cluster(1)
cluster_name, similarities = sqc.infer_cluster_name(some_cluster)

# naming all clusters and dumping results to a csv file based on model and clustering parameters 
sqc.name_all_clusters(dump_csv=True)

# save a DataFrame of scored cluster and also dump to csv
scored = sqc.score_all_clusters(dump_csv=True)
```

#### 6.2 Daily clustering

This is most probable scenario is where data of multiple days is being passed to the class constructor as a csv file. Since this is the case, `make_daily_clusters()` is used, and it conveniently includes all the preprocessing methods as well the clustering itself <u>for each day separately</u> .

##### 6.2.1 Runs as a part of the pipeline (i.e, always)

Inside `make_daily_clusters()` the following methods will be executed automatically:

##### Loading and preprocessing

```python
aggregate_df()
clean_search_queries()
remove_i_will()
```

##### Clustering

```python
make_clusters()
```

##### Scoring and Naming

This will be executed if `make_daily_clusters()` argument `score=True`, which is default:

```python
score_all_clusters()
```

*Note: the naming method `name_all_clusters()` is called from inside `score_all_clusters()`.*

##### 6.2.2 Analysis - not part of the pipeline, needs to be called from `main()`

Once `make_daily_clusters()` is done, there are a few handy out-fo-the-box functions that can be used to analyse the results:

* Use `create_daily_scores_table()` to create a DataFrame in which you can view scores across time. 
* Use `plot_daily_scores(['query a', 'query b', 'query c'])` to show a line plot of clusters scores over days for some specified queries. 
* Use `plot_n_random_scores(n=5)` to show 5 random clusters scores over days.  

### 7. Future Versions

* Creating a Cluster class that would hanle all Cluster methods and properties (at the moment, everything is being handled by the SearchQueriesClustering, the clusterer class)
* Pythonize the module (private methods to be preceded with a _, etc.)
* Addition of other clustering algorithms (DBScan, Agglomarative Clustering to name a couple).
* Optimal parameters for Doc2Vec and the clustering algorithm (probably HDBScan)
  * Testing different [TaggedDocument](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.TaggedDocument) approaches
  * Understanding interaction of doc2vec embedding matrix size and content and how it affect the different clustering algorithms. 
* Further improving both cluster naming and scoring in order to obtain more stable clusters over days. 

