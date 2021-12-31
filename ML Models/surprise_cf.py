import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD,accuracy, NMF, KNNWithMeans, KNNBaseline, CoClustering, KNNWithZScore, BaselineOnly, NormalPredictor, dump
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold

def recommendation(user_id):
    user = book.copy()
    already_read = books_data[books_data['user_id'] == user_id]['book_id'].unique()
    user = user.reset_index()
    user = user[~user['book_id'].isin(already_read)]
    user['Estimate_Score']=user['book_id'].apply(lambda x: algo.predict(user_id, x).est)
    user = user.drop('book_id', axis = 1)
    user = user.sort_values('Estimate_Score', ascending=False)
    print(user.head(10))
    return user.head(10)


books = pd.read_csv('books.csv')
book = books[['book_id','authors','title']]
ratings = pd.read_csv('ratings.csv')
books_data = pd.merge(book, ratings, on='book_id')

reader = Reader(rating_scale=(1,5))


data = Dataset.load_from_df(ratings[['book_id', 'user_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25)

algo = BaselineOnly()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions, verbose=True)

dump.dump('surprise_cf.pickle', predictions=None, algo=algo, verbose=1)

# loading the dumped file and recommending...
red2 = recommendation(314)

print(red2)
_, algo1 = dump.load("surprise_cf.pickle")


