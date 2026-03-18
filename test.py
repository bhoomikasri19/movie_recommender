from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import os
import ast


#Load dataset
df=pd.read_csv("/Users/bhoomikasrivastava19/Documents/movie_recommender/tmdb_5000_movies.csv")

df=df[["title","overview","genres","keywords","popularity"]]

#handle missing data
print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())

df=df.sort_values("popularity",ascending=False).head(2000)
df=df.reset_index(drop=True)

# print(df.shape)
# print(df.head(3))
# print(df.iloc[0])



#cleaning json data to python using ast
df['genres']=df["genres"].apply(lambda x:[i["name"]for i in ast.literal_eval(x)])
df["keywords"]=df["keywords"].apply(lambda x:[i["name"]for i in ast.literal_eval(x)])

print(df.head(2))
print(df.iloc[0])

#converting list to string
df["genres"]=df["genres"].apply(lambda x:" ".join(x))
df["keywords"]=df["keywords"].apply(lambda x:" ".join(x))

df=df.drop(columns=['popularity'])

#creating a tag that contains overview,genres,keywords
df['tags']=df['overview']+" "+df["genres"]+" "+df["keywords"]

df=df[['title','tags']]
print(df.head(2))

#convert text to num which dont have any category using count vect which convert [1,0,0,1]
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(df['tags'])
print(vectors.shape)

#calculate similarity between movie
similarity=cosine_similarity(vectors)
print(similarity.shape)

#User types a movie name → "Avatar"
#Find Avatar's index in the dataframe → row 0
#Get Avatar's row from similarity table → [1.0, 0.3, 0.1, ...]
#Sort by highest similarity score
#Return top 5 movie titles

def recommend(movie):
    index=df[df['title']==movie].index[0]
    distance=similarity[index]
    movie_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(df.iloc[i[0]].title)

# recommend("The Dark Knight")

#saving the model
os.makedirs("model",exist_ok=True)
joblib.dump(df,"model/movies.pkl")
joblib.dump(similarity,"model/similarity.pkl")
print("model saved!")