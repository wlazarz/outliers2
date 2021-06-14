def fill_empty(X, cat_columns = []):
    X = X.apply(lambda x:x.fillna(x.value_counts().index[0]))
    for i in X:
        max_col=X[i].value_counts().index[0]
        if max_col=='?': 
            max_col=X[i].value_counts().index[1]
        X[i] = X[i].replace(['?'],max_col)
        if i not in cat_columns and (X[i].dtype==object or X[i].dtype==str):
            X[i] = [x.replace(',', '.') for x in X[i]]
            X = X.astype({i: 'float64'})
    return X

def drop_empty(X):
    rows_count = X.shape[0]
    for i in X:
        if df[df[i] == '?'].shape[0]>0.6*rows_count:
            X.drop(i, axis=1, inplace=True)
    return X

def variable_coding(X, columns):
    val_dict = {}
    i = 1
    for col in columns:
        uniq_val = X[col].value_counts().keys().tolist()
        for val in uniq_val:
            X.loc[X[col] == val, col] = i
            val_dict[(col, i)] = val
            i += 1
    return X, val_dict

def variable_encoding(X, val_dict):
    for key in val_dict:
        column_name=key[0]
        X.loc[X[column_name] == key[1], column_name] = val_dict[key]
    return X

