#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[18]:


import pickle
import streamlit as st
import spacy
from bs4 import BeautifulSoup
import re
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer


# In[19]:


def lemmatize_words(text):
        lemma = nltk.WordNetLemmatizer()
        words = text.split()
        words = [lemma.lemmatize(word,pos='v') for word in words]
        return ' '.join(words) 


# In[20]:


def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()
def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)
def remove_characters(text):
    return re.sub("[^a-zA-Z]"," ",text)


# In[36]:


def cleaning(text):
    text = remove_html(text)
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    text = lemmatize_words(text)  
    return text

clean = lambda x: cleaning(x)


# In[6]:





# In[21]:


tf_idf_converter=pickle.load(open('C:/Users/lenvo/Desktop/DATA SCIENCE/Project/Sentiemntal/tf_idf_converter.pkl','rb'))


# In[22]:


model=pickle.load(open('C:/Users/lenvo/Desktop/DATA SCIENCE/Project/Sentiemntal/classifier.pkl','rb'))


# In[39]:


def news_prediction(sample):
    Result=model.predict(tf_idf_converter.transform(sample))
    if (Result[0]==0):
        return'The news is FAKE'
    else:
        return 'The news is TRUE'
    

    
         


# In[40]:


def main():
    st.title('FAKE NEWS AND TRUE NEWS DETECTION')
    st.subheader('LETS INVESTIGATE AND GET RESULT')
    
    if st.checkbox("Show text after cleaning "):
        st.subheader("cleaned text")
        text=st.text_area("Enter your text","Type Here")
        if st.button("Analyze"):
            nlp_result=cleaning(text)
            st.success(nlp_result)
    if st.checkbox("Converted text to TFIDF VEC "):
        st.subheader("TFIDF VECTORIZER")
        text=st.text_area("Enter your text","Type Here")
        if st.button("Convert"):
            sample=[text]
            TFIDF_result=tf_idf_converter.transform(sample)
            st.success(TFIDF_result)
    if st.checkbox("Evaluate the NEWS "):
        text=st.text_area("Enter Any News Headline","Type Here")
        if st.button("Check"):
            if len(text) < 1:
                st.write(" ")
            else:
                nlp_result=cleaning(text)
                sample=[nlp_result]
                Result=news_prediction(sample)
                st.success(Result)
        
     
    
    
    
if __name__=='__main__':
    main()


# In[ ]:




