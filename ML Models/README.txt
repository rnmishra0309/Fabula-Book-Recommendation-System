################## Content_based.py ######################

This python file contains all the codes to train your dataset based on
content based filtering methods.

In this file, we have three user defined functions for recommendations:

1. authors_recommendations(title): After creating and training the model, 
   this function takes a book's title as an argument and provides recommendations based on the author's name.

2. tags_recommendations(title): After creating and training the model, 
   this function takes a book's title as an argument and provides       recommendations based on the genre of the book.

3. corpus_recommendations(title): After creating and training the model, 
   this function takes a book's title as an argument and provides       recommendations based on the author's name as well as the genre of the book.




################# surprise_cf.py ######################

This python file contains all the codes to train your dataset based on
collaborative based filtering methods.

In this file, we used a third party library known as 'scikit-surprise'
for training the models based on SVD method.

This has one user defined function :

1. recommendation(user_id): This function is responsible for obtaining
the recommendations from the model when a user's id is passed.