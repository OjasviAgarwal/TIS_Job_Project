# -*- coding: utf-8 -*-

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
        self.jobs_info_df = pd.DataFrame(jobs_list)
        linkedin_df = linkedin_scrapper.get_linkedin_jobs_info()
        self.jobs_info_df = self.jobs_info_df.append(linkedin_df, ignore_index=True)
          
    def keywords_extract(self, text): 
        text = re.sub("[^a-zA-Z+3]"," ", text) 
        text = text.lower().split()
        stops = set(stopwords.words("english")) 
        text = [w for w in text if not w in stops]
        text = list(set(text))
        keywords = [str(word) for word in text if word in overall_dict]
        return keywords
 
    def keywords_count(self, keywords, counter): 
        print("Inside the keywords_count function")
        print("Printing the keywords coming from skill disctionary")   
        print(keywords)
        print("Printing the counter we have passed form resume")
        print(counter)        
        keyword_count = pd.DataFrame(columns = ['Freq'])
        print(keyword_count)
        for each_word in keywords: 
            keyword_count.loc[each_word] = {'Freq':counter[each_word]}
        return keyword_count

    def get_cosine_similarity_bit_vector(self,x,y):
        l1=[]
        l2=[]
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

        for i in range(len(rvector)):
            c+= l1[i]*l2[i]
        cosine = c / float((sum(l1)*sum(l2))**0.5) 
        return cosine
    
    
    def cal_similarity(self, resume_keywords, location=None):     
        num_jobs_return = 5
        similarity_cosine = []
        j_info = self.jobs_info_df.loc[self.jobs_info_df['location']==location].copy() if len(location)>0 else self.jobs_info_df.copy()
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
        self.jobs_info_df['keywords'] = [self.keywords_extract(job_desc) for job_desc in self.jobs_info_df['desc']]
        
    def extract_resume_keywords(self, resume_pdf): 
        resume_file = open(resume_pdf, 'rb')
        resume_reader = PyPDF2.PdfFileReader(resume_file)
        resume_content = [resume_reader.getPage(x).extractText() for x in range(resume_reader.numPages)]
        resume_keywords = [self.keywords_extract(page) for page in resume_content]
        print("Printing the keywords extracted from the resume")
        print(resume_keywords)
        resume_freq = Counter()
        f = [print(resume_freq.update(item)) for item in resume_keywords]
        print("Here we have creater counts of keywords for all the lists extracted based on each page")
        print(resume_freq)
        resume_skills = self.keywords_count(overall_skills_dict, resume_freq)
        return(resume_skills[resume_skills['Freq']>0])
