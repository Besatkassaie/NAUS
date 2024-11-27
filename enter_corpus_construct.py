import constructNoveltyGroundTruth



import re




args2 = ['--encoder', 'cl', '--benchmark', 'santos', 
        '--augment_op', 'drop_col','--sample_meth', 'tfidf_entity',
        '--matching', 'exact',   '--table_order', 'column',
        '--run_id', '0' ,   '--K','10' , '--threshold', '0.7', '--penalty', '1', '--tokenize','1', '--bot', '1', 
        '--penalty_degree', '1', '--late_penalty', '1', '--entropy', '1']

constructNoveltyGroundTruth.main(args2)