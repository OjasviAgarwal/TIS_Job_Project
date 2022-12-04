# -*- coding: utf-8 -*-
"""
Tokenize text, extract keywords, and recommend jobs by matching keywords from resume with jobs
"""
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter 
import pandas as pd
import PyPDF2
import config
import linkedin_scrapper
#We need to update our skill list based on what dataset we decide to finalise
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
        #calling the linkedIn scrapper to get that added to the original dataframe
        linkedin_df = linkedin_scrapper.get_linkedin_jobs_info()
        #print(linkedin_df)
        #print(len(self.jobs_info_df.index))
        #print(len(linkedin_df.index))
        #self.jobs_info_df.concat(linkedin_df)
        #self.jobs_info_df.concat([self.jobs_info_df, linkedin_df])
        self.jobs_info_df = self.jobs_info_df.append(linkedin_df, ignore_index=True)
        #print(len(self.jobs_info_df.index))
        #print(self.jobs_info_df.tail(10))
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

    def get_cosine_similarity_bit_vector(self,x,y):
        l1=[]
        l2=[]
        #rvector = x.union(y)
        #print(len(y) == len(set(y)))
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
        #print("similarity: ", cosine)
        return cosine
    
    
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
        similarity_cosine = []
        j_info = self.jobs_info_df.loc[self.jobs_info_df['location']==location].copy() if len(location)>0 else self.jobs_info_df.copy()
        #if number of rows in the dataframe are less than 5, then we ll update 5 to the number of rows in the dataframe
        if j_info.shape[0] < num_jobs_return:        
            num_jobs_return = j_info.shape[0]  
        for job_skills in j_info['keywords']:
            similarity_cosine.append(self.get_cosine_similarity_bit_vector(resume_keywords.tolist(),job_skills))
        j_info['similarity_cosine'] = similarity_cosine
        print(j_info)
        top_match_based_on_cosine = j_info.sort_values(by='similarity_cosine', ascending=False).head(num_jobs_return)        
        print(top_match_based_on_cosine)
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
        #print(resume_skills[resume_skills['Freq']>0])
        return(resume_skills[resume_skills['Freq']>0])
