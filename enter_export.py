import test_naive_search_export
import logging



logging.basicConfig(level=logging.DEBUG)

# we run this to export result of native starmie 



args2 = ['--encoder', 'cl', '--benchmark', 'ugen_v2/ugenv2_small', 
        '--augment_op', 'drop_col','--sample_meth', 'tfidf_entity',
        '--matching', 'exact',   '--table_order', 'column',
        '--run_id', '0' ,   '--K','20', '--threshold', '0.7', '--restrict', '1']

test_naive_search_export.main(args2)