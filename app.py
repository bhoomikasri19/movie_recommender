import streamlit as st
import joblib

#load file
df=joblib.load("model/movies.pkl")
similarity=joblib.load("model/similarity.pkl")

#function
def recommend(movie):
    index=df[df['title']==movie].index[0]
    distance=similarity[index]
    movie_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        st.write(df.iloc[i[0]].title)

st.title("Movie Recommendation system")

selected_movie=st.selectbox("Select a Movie",df['title'].values)

if st.button("Recommend"):
    st.subheader("Top 5 recommendation based on your selected movie:")
    results=recommend(selected_movie)

