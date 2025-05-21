# Table Union Search with Penalized Approach

This repository implements a novel penalized search approach for table union search, which enhances the discovery of unionable tables by incorporating both semantic and lexical similarity measures.

## Overview

The penalized search approach combines two key components:
1. **Semantic Similarity**: Uses vector embeddings to capture semantic relationships between tables
2. **Lexical Distance**: Measures the lexical similarity between table contents

The approach penalizes the unionability score based on these measures to promote diversity in search results while maintaining relevance.

## Key Features

- **Hybrid Scoring**: Combines semantic and lexical measures for more accurate table union discovery
- **Configurable Penalty**: Adjustable penalty degree (p_degree) to control the balance between diversity and relevance
- **Efficient Processing**: Optimized implementation using pandas and numpy for fast computation
- **Flexible Integration**: Can be used with various table embedding methods (e.g., Starmie, TF-IDF)

## Implementation Details

The penalized search approach works as follows:

1. **Initial Ranking**: Uses an initial ranking (e.g., from Starmie) to identify potential unionable tables
2. **Column Alignment**: Processes column alignments between query and data lake tables
3. **Score Computation**:
   - Calculates semantic similarity using vector embeddings
   - Computes lexical distance between table contents
   - Applies penalty based on the formula: `(lexical_distance ^ p_degree) * semantic_similarity`
4. **Result Ranking**: Ranks tables based on the penalized scores

## Usage

```python
# Initialize the penalized search
penalize_search = Penalized_Search(
    dsize=20,  # Domain size threshold
    dataFolder="path/to/data",
    table_path="path/to/vectors",
    query_path_raw="path/to/queries",
    table_path_raw="path/to/tables",
    processed_path="path/to/processed",
    index_file_path="path/to/index"
)

# Load required data
penalize_search.load_column_alignment_data(alignment_file)
penalize_search.load_unionable_tables(initial_ranking_file)
all_vectors = penalize_search.load_starmie_vectors(dl_vectors, query_vectors)

# Perform search
results = penalize_search.perform_search_optimized(
    p_degree=1,  # Penalty degree
    k=10,        # Number of results
    all_vectors=all_vectors
)
```

## Evaluation

The approach can be evaluated using various metrics:
- Precision and Recall at different k values
- Syntactic Novelty Measure (SNM)
- Execution time analysis
- Diversity metrics

## Dependencies

- Python 3.x
- pandas
- numpy
- scipy
- SetSimilaritySearch
- pickle5

## Citation

If you use this code in your research, please cite:
```

```
