# Embedding Class

An embedding class that takes in descriptions for POIs and generate corresponding embeddings.

## Table of Contents

- [Usage](#usage)
- [Contributing](#contributing)

## Usage

1. You can initialize an embedding class without initializing any parameters.
2. In order to generate embeddings, you need to pass in your full dataframe with poi names, descriptions, and tags into self.output_embedding. In order to extract information from the dataframe, 
you need to provide the names of the columns you are going to use.
3. In order to get visualization and k-means (n=5) evaluation of the embeddings, you may use the
self.get_evaluation function. You need to pass in your full dataframe with poi names, descriptions, tags, and embeddings you have generated. And you also need the three tags you pick from the most 
frequently appeared tags.
4. The results will be saved in the "output" folder.
5. NOTE (!!!): SOME OF THE ENCODINGS IN encodings.pickle ARE [0.0, 0.0, ...], WHICH MEANS THAT THE ORIGINAL dESCRIPTIONS ARE NONE.
    ### Run
    
        To run the script, use the following command:
        
        ```bash
        python EMBEDDING_CLASS.py
        ```

## Contributing

Others can contribute to the project by providing advice.