# -*- coding: utf-8 -*-
"""
Build a job-description and skill-keyword based job recommendation engine, 
which would match keywords from resume to jobs in ____________.
Step 1: Scrape "________________" jobs from __________
Step 2: Tokenize and extract skill keywords from job descriptions
Step 3: Tokenize and extract skill keywords from resume
Step 4: Calculate Cosine similarity of keywords from posted jobs and resume, 
        and recommend top 5 matches 
"""
import sys 
import config, web_scrapper
#importing another python file called skill_keyword_match
from skill_keyword_match import skill_keyword_match
import nltk
nltk.download('stopwords')


def main():
    # If city included, only search and recommend jobs in the city
    location = ''
    if (len(sys.argv) > 1):
        # Check if input city name matches our pre-defined list
        if (sys.argv[1] in config.JOB_LOCATIONS):
            location = sys.argv[1]
        else:
            sys.exit('*** Please try again. *** \nEither leave it blank or input a city from this list:\n{}'.format('\n'.join(config.JOB_LOCATIONS)))
    # ---------------------------------------------------
    # ---- Scrape from web or read from local saved -----
    # ---------------------------------------------------
    #you are going to the web_scrapper.py file and calling the get_jobs_info class
    jobs_info = web_scrapper.get_jobs_info(location)
    #print(jobs_info[0])
    #print(jobs_info)  #this is throwing a lot of text data  
    # ---------------------------------------------------
    # -------- Keyword extraction and analysis ----------
    # ---------------------------------------------------
    # Initialize skill_keyword_match with job_info
    skill_match = skill_keyword_match(jobs_info)
    # Extract skill keywords from job descriptions 
    #This will remain static for all the runs 
    skill_match.extract_jobs_keywords()
    # ---------------------------------------------------
    # -- Job recommendation based on skill matching -----
    # ---------------------------------------------------
    resume_skills = skill_match.extract_resume_keywords(config.SAMPLE_RESUME_PDF)
    #print(resume_skills)
    #print(resume_skills)
    # Calculate similarity of skills from a resume and job posts 
    top_job_matches = skill_match.cal_similarity(resume_skills.index, location)
    # Save matched jobs to a file
    top_job_matches.to_csv(config.RECOMMENDED_JOBS_FILE+location+'.csv', index=False)
    print('File of recommended jobs saved')

if __name__ == "__main__": 
    main()

