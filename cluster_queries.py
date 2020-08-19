from search_queries_clustering import *
from eda_utils import *
from collections import OrderedDict

DATA_FILE = 'bq-results-20200713-172255-kpavmmc9bgkk.csv'
BIG_DATA_FILE = 'bq-results-20200731-164149-aypit9tb267h.csv'
CLUSTERING_METHOD = 'hdbscan'

if __name__ == '__main__':

    ######################
    # general clustering #
    ######################

    # cleaning directory on startup (use carefully)
    if config['global']['drastic_cleanup']:
        SearchQueriesClustering.drastic_cleanup()

    # crating an instance and loading the data
    # sqc = SearchQueriesClustering(DATA_FILE)
    #

    #
    # sqc.aggregate_df()
    # sqc.clean_search_queries()
    # sqc.remove_i_will()
    #
    # # crating a doc2vec embedding and performing clustering of choice
    # labeled_queries, clustering_obj = sqc.make_clusters(algo=CLUSTERING_METHOD)
    #
    # # saving model details to file
    # sqc.document_process_details()

    # showing some clusters
    # sqc.show_some_clusters()
    #
    # # fetching a cluster as a dict
    # sqc.get_cluster_as_dict(5)
    #
    # # clustering with different parameters
    # sqc.make_clusters(algo='hdbscan', min_cluster_size=15)
    #
    # # manual inference of cluster name
    # some_cluster = sqc.get_cluster(1)
    # cluster_name, similarities = sqc.infer_cluster_name(some_cluster)
    #
    # sqc.name_all_clusters(dump_csv=True)

    # scored = sqc.score_all_clusters(dump_csv=True)



    ####################
    # Daily Clustering #
    ####################

    # Execution

    sqc_daily = SearchQueriesClustering(BIG_DATA_FILE)
    sqc_daily.make_daily_clusters()
    scores_table = sqc_daily.create_daily_scores_table()


    # Analytics examples

    sqc_daily.plot_n_random_scores()

    some_clusters = ['marketing_digital', 'podcast_promotion', 'data_entry', 'logo_art']
    sqc_daily.plot_daily_scores(some_clusters)

    sqc_daily.plot_daily_scores(['website', 'twitch', 'dropshipping', 'crowdfunding', 'matlab'])

    logo_clusters = pd.DataFrame(sqc_daily.get_available_cluster_names())[0]
    logo_clusters = logo_clusters[logo_clusters.str.find('logo') != -1].to_list()
    sqc_daily.plot_daily_scores(np.random.choice(logo_clusters, 5))

    sqc_daily.plot_daily_scores(['sport_logo', 'channel_logo', 'farm_logo', 'designing_logo', 'minimalist_logo', 'banner_logo'])

    logo_daily_scores = sqc_daily.get_cluster_daily_scores('minimalist_logo')


    sqc_daily.plot_n_random_scores()


    sqc_daily.plot_daily_scores(['tiktok', 'youtube', 'facebook', 'instagram'])

    sqc_daily.get_cluster_daily_queries('seo')

    sqc_daily.plot_daily_scores(['logo', 'logo_design','design'])



    # need to implement in class
    def show_cluster_queries(cluster_name):
        daily_queries = sqc_daily.get_cluster_daily_queries(cluster_name)
        for date, query in daily_queries.items():
            print(date)
            for q in query:
                q = sorted(q.items(), key=lambda kv: kv[1], reverse=True)
                pprint(q)
    show_cluster_queries('instagram')