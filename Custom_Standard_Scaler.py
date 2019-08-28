### Classes Custom Standard_Scaler  

class Standard_Scaler(BaseEstimator,TransformerMixin):
    
    def __init__(self, X, text_columns = None, numerical_columns= None):
        self.scaler = StandardScaler()
        self.text_columns = text_columns
        self.numerical_columns = numerical_columns
    
    def fit(self, X, *args):
        self.numerical_columns = X.select_dtypes(include= [np.number])
        self.text_columns = X[X.columns.difference(list(self.numerical_columns.columns))]
        self.scaler = self.scaler.fit(self.numerical_columns)
        return self
    
    
    def transform(self, X, *args):
        self.text_columns = X[X.columns.difference(list(self.numerical_columns.columns))]
        self.numerical_columns = X.select_dtypes(include= [np.number])
        # Fitting & Transforming The 
        numerical = pd.DataFrame(self.scaler.transform(self.numerical_columns), columns = X.columns.difference(['Article_Text']),
                                index = self.numerical_columns.index)
        
        # Horizontally Stacking The Scaled Numerical Data With The Text Data
        X = pd.concat([numerical, self.text_columns], axis=1, sort=True)
        
        return X
