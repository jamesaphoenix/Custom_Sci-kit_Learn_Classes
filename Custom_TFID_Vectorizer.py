class TFID_Vectorizer(BaseEstimator,TransformerMixin):
    
    def __init__(self, X, ngram_range = None, min_df = None, max_df = None, 
                 max_features= None):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.tvec = TfidfVectorizer(stop_words='english', ngram_range = self.ngram_range, min_df= self.min_df, 
                                    max_df = self.max_df, max_features = self.max_features)
    
    def fit(self, X, *args):
        self.text_columns = X.select_dtypes('object')['Article_Text']
        self.tvec = self.tvec.fit(self.text_columns)
        return self
    
    
    def transform(self, X, *args):
        self.text_columns = X.select_dtypes('object')['Article_Text']
        self.numerical_columns = X[X.columns.difference(['Article_Text'])]
        
        X = self.tvec.transform(self.text_columns)
        self.sparse_numerical_data = sparse.csr_matrix(self.numerical_columns)
        
        #Concatenating The Sparse Matrices Together
        X = sparse.hstack([X, self.sparse_numerical_data])
        
        return X
