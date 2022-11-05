# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:52:03 2018

@author: Peng Wang

Tokenize text, extract keywords, and recommend jobs by matching keywords from resume with jobs
"""
import numpy as np
import re
from nltk.corpus import stopwords
from collections import Counter 
import pandas as pd
import PyPDF2
import config
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy
# The following data science skill sets are modified from 
# https://github.com/yuanyuanshi/Data_Skills/blob/master/data_skills_1.py
program_languages = ['bash','r','python','java','c++','ruby','perl','matlab','javascript','scala','php']
analysis_software = ['excel','tableau','sas','spss','d3','saas','pandas','numpy','scipy','sps','spotfire','scikit','splunk','power','h2o']
ml_framework = ['pytorch','tensorflow','caffe','caffe2','cntk','mxnet','paddle','keras','bigdl']
bigdata_tool = ['hadoop','mapreduce','spark','pig','hive','shark','oozie','zookeeper','flume','mahout','etl']
ml_platform = ['aws','azure','google','ibm']
methodology = ['agile','devops','scrum']
databases = ['sql','nosql','hbase','cassandra','mongodb','mysql','mssql','postgresql','oracle','rdbms','bigquery']
overall_skills_dict = program_languages + analysis_software + ml_framework + bigdata_tool + databases + ml_platform + methodology
education = ['master','phd','undergraduate','bachelor','mba']
overall_dict = overall_skills_dict + education
jobs_info_df = pd.DataFrame()

class skill_keyword_match:
    def __init__(self, jobs_list):
        '''
        Initialization - converts list to DataFrame
        Input: 
            jobs_list (list): a list of all jobs info
        Output: 
            None
        '''
        self.jobs_info_df = pd.DataFrame(jobs_list)
        #print(self.jobs_info_df)
          
    def keywords_extract(self, text): 
        '''
        Tokenize webpage text and extract keywords
        Input: 
            text (str): text to extract keywords from
        Output: 
            keywords (list): keywords extracted and filtered by pre-defined dictionary
        ''' 
        #We are passing every value of job description as text in this function       
        # Remove non-alphabet; 3 for d3.js and + for C++
        text = re.sub("[^a-zA-Z+3]"," ", text) 
        text = text.lower().split()
        stops = set(stopwords.words("english")) #filter out stop words in english language
        text = [w for w in text if not w in stops]
        text = list(set(text))
        # We only care keywords from the pre-defined skill dictionary
        keywords = [str(word) for word in text if word in overall_dict]
        return keywords
 
    def keywords_count(self, keywords, counter): 
        '''
        Count frequency of keywords
        Input: 
            keywords (list): list of keywords
            counter (Counter)
        Output: 
            keyword_count (DataFrame index:keyword value:count)
        '''
        print("Inside the keywords_count function")
        print("Printing the keywords coming from skill disctionary")   
        print(keywords)
        print("Printing the counter we have passed form resume")
        print(counter)        
        keyword_count = pd.DataFrame(columns = ['Freq'])
        print(keyword_count)
        for each_word in keywords: 
            keyword_count.loc[each_word] = {'Freq':counter[each_word]}
            #print(keyword_count.loc[each_word]
        #Here we are storing the frequency of the resume keywords with respect to the overall generic skill
        #dictionary (so freq = 0 means that it is not present in resume keywords else we put the freq count from resume keyword)
        #print(keyword_count)
        return keyword_count
    
    def exploratory_data_analysis(self):
        '''
        Exploratory data analysis
        Input: 
            None
        Output: 
            None
        '''         
        # Create a counter of keywords
        doc_freq = Counter() 
        f = [doc_freq.update(item) for item in self.jobs_info_df['keywords']]
        
        # Let's look up our pre-defined skillset vocabulary in Counter
        overall_skills_df = self.keywords_count(overall_skills_dict, doc_freq)
        # Calculate percentage of required skills in all jobs
        overall_skills_df['Freq_perc'] = (overall_skills_df['Freq'])*100/self.jobs_info_df.shape[0]
        overall_skills_df = overall_skills_df.sort_values(by='Freq_perc', ascending=False)  
        # Make bar plot 
        plt.figure(figsize=(14,8))
        overall_skills_df.iloc[0:30, overall_skills_df.columns.get_loc('Freq_perc')].plot.bar()
        plt.title('Percentage of Required Data Skills in Data Scientist/Engineer/Analyst Job Posts')
        plt.ylabel('Percentage Required in Jobs (%)')
        plt.xticks(rotation=30)
        plt.show()
        
        # Plot word cloud
        all_keywords_str = self.jobs_info_df['keywords'].apply(' '.join).str.cat(sep=' ')        
        # lower max_font_size, change the maximum number of word and lighten the background:
        wordcloud = WordCloud(background_color="white").generate(all_keywords_str)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
         
        # Let's look up education requirements
        education_df = self.keywords_count(education, doc_freq)
        # Merge undergrad with bachelor
        education_df.loc['bachelor','Freq'] = education_df.loc['bachelor','Freq'] + education_df.loc['undergraduate','Freq'] 
        education_df.drop(labels='undergraduate', axis=0, inplace=True)
        # Calculate percentage of required skills in all jobs
        education_df['Freq_perc'] = (education_df['Freq'])*100/self.jobs_info_df.shape[0] 
        education_df = education_df.sort_values(by='Freq_perc', ascending=False)  
        # Make bar plot 
        plt.figure(figsize=(14,8))
        education_df['Freq_perc'].plot.bar()
        plt.title('Percentage of Required Education in Data Scientist/Engineer/Analyst Job Posts')
        plt.ylabel('Percentage Required in Jobs (%)')
        plt.xticks(rotation=0)
        plt.show()
        
        # Plot distributions of jobs posted in major cities 
        plt.figure(figsize=(8,8))
        self.jobs_info_df['location'].value_counts().plot.pie(autopct='%1.1f%%', textprops={'fontsize': 10})
        plt.title('Data Scientist/Engineer/Analyst Jobs in Major Canadian Cities \n\n Total {} posted jobs in last {} days'.format(self.jobs_info_df.shape[0],config.DAY_RANGE))
        plt.ylabel('')
        plt.show()
    
        
    '''
    def loadGloveModel(self):
        gloveFile = "data/glove.6B.50d.txt"
        print ("Loading Glove Model")
        with open(gloveFile, encoding="utf8" ) as f:
            content = f.readlines()
        model = {}
        for line in content:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print ("Done.",len(model)," words loaded!")
        self.model = model
    '''
    
    def cosine_distance_wordembedding_method(self,x, y):
        vector_1 = np.mean([self.model[word] for word in (x)],axis=0)
        vector_2 = np.mean([self.model[word] for word in (y)],axis=0)
        cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
        print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
        return cosine
    
    def get_jaccard_sim(self, x_set, y_set): 
        '''
        Jaccard similarity or intersection over union measures similarity 
        between finite sample sets,  and is defined as size of intersection 
        divided by size of union of two sets. 
        Jaccard calculation is modified from 
        https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
        Input: 
            x_set (set)
            y_set (set)
        Output: 
            Jaccard similarity score
        '''         
        intersection = x_set.intersection(y_set)
        return float(len(intersection)) / (len(x_set) + len(y_set) - len(intersection))

    def get_vectors(self,v):
        #print("Hi")
        #text = [t for t in v]
        text = v
        #print(v)
        #print(text)
        vectorizer = CountVectorizer()
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()

    def get_cosine_similarity_bit_vector(self,x,y):
        l1=[]
        l2=[]
        #rvector = x.union(y)
        print(len(y) == len(set(y)))
        if(len(x) == 0 or len(y) == 0):
            cosine = 0
            return cosine
        rvector = list(set().union(x,y))
        for w in rvector:
            if w in x: l1.append(1) # create a vector
            else: l1.append(0)
            if w in y: l2.append(1)
            else: l2.append(0)
        c = 0 

        # cosine formula 
        for i in range(len(rvector)):
            c+= l1[i]*l2[i]
        cosine = c / float((sum(l1)*sum(l2))**0.5) #Here we are not squaring because we have 0/1 values in vector
        print("similarity: ", cosine)
        return cosine

    def get_cosine_sim(self, x, y): 
        '''
        https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
        Input: 
            x (list)
            y (list)
        Output: 
            Cosine similarity score
        '''         
        #vectors = [t for t in get_vectors(*strs)]
        #print("1")
        #print(type(x))
        vector_x = [t for t in self.get_vectors(x)]
        vector_y = [t for t in self.get_vectors(y)]
        print(vector_x)
        print(vector_y)
        print(cosine_similarity(vector_x,vector_y))
        return cosine_similarity(vector_x,vector_y)

    
    
    def cal_similarity(self, resume_keywords, location=None):
        '''
        Calculate similarity between keywords from resume and job posts
        Input: 
            resume_keywords (list): resume keywords
            location (str): city to search jobs
        Output: 
            top_match (DataFrame): top job matches
        '''         
        num_jobs_return = 5
        similarity = []
        similarity_cosine = []
        similarity_cosine_glove = []
        #self.loadGloveModel()
        j_info = self.jobs_info_df.loc[self.jobs_info_df['location']==location].copy() if len(location)>0 else self.jobs_info_df.copy()
        #print(j_info)
        #print(j_info.shape[0])
        #if number of rows in the dataframe are less than 5, then we ll update 5 to the number of rows in the dataframe
        if j_info.shape[0] < num_jobs_return:        
            num_jobs_return = j_info.shape[0]  
        for job_skills in j_info['keywords']:
            #print(job_skills)
            #similarity.append(self.get_jaccard_sim(set(resume_keywords), set(job_skills)))
            #print(type(resume_keywords))
            #print(set(resume_keywords))
            #print(type(job_skills))
            #similarity_cosine.append(self.get_cosine_sim(resume_keywords.tolist(),job_skills))
            similarity_cosine.append(self.get_cosine_similarity_bit_vector(resume_keywords.tolist(),job_skills))
            #similarity_cosine_glove.append(self.cosine_distance_wordembedding_method(resume_keywords.tolist(),job_skills))
        #j_info['similarity'] = similarity
        j_info['similarity_cosine'] = similarity_cosine
        #j_info['similarity_cosine_glove'] = similarity_cosine_glove
        #print(j_info['similarity'])
        #print(j_info)
        #top_match = j_info.sort_values(by='similarity', ascending=False).head(num_jobs_return)
        top_match_based_on_cosine = j_info.sort_values(by='similarity_cosine', ascending=False).head(num_jobs_return)        
        #top_match_based_on_cosine_glove = j_info.sort_values(by='similarity_cosine_glove', ascending=False).head(num_jobs_return)
        # Return top matched jobs
        #print(top_match_based_on_cosine_glove)
        print(top_match_based_on_cosine)
        #return top_match
        return top_match_based_on_cosine
      
    def extract_jobs_keywords(self):
        '''
        Extract skill keywords from job descriptions and add a new column 
        Input: 
            None
        Output: 
            None
        ''' 
        self.jobs_info_df['keywords'] = [self.keywords_extract(job_desc) for job_desc in self.jobs_info_df['desc']]
        #print(self.jobs_info_df['keywords'])
        
    def extract_resume_keywords(self, resume_pdf): 
        '''
        Extract key skills from a resume 
        Input: 
            resume_pdf (str): path to resume PDF file
        Output: 
            resume_skills (DataFrame index:keyword value:count): keywords counts
        ''' 
        # Open resume PDF
        resume_file = open(resume_pdf, 'rb')
        # creating a pdf reader object
        resume_reader = PyPDF2.PdfFileReader(resume_file)
        # Read in each page in PDF
        resume_content = [resume_reader.getPage(x).extractText() for x in range(resume_reader.numPages)]
        # Extract key skills from each page
        #We are getting 2 lists of resume keywords for 2 pages
        resume_keywords = [self.keywords_extract(page) for page in resume_content]
        print("Printing the keywords extracted from the resume")
        print(resume_keywords)
        # Count keywords
        resume_freq = Counter()
        #print(resume_freq)
        #Here we are getting count with respect to keywords in the resume itself 
        f = [print(resume_freq.update(item)) for item in resume_keywords]
        #print("Helooooo", f)
        #Here we have creater counts of keywords for all the lists extracted based on each page 
        print("Here we have creater counts of keywords for all the lists extracted based on each page")
        print(resume_freq)
        #print(type(f))
        #print(resume_freq)
        # Get resume skill keywords counts
        #Here we are comparing th
        resume_skills = self.keywords_count(overall_skills_dict, resume_freq)
        #print(resume_skills)
        print(resume_skills[resume_skills['Freq']>0])
        return(resume_skills[resume_skills['Freq']>0])
