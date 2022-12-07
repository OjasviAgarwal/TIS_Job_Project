# -*- coding: utf-8 -*-
import sys 
import config, web_scrapper
from skill_keyword_match import skill_keyword_match
import nltk
nltk.download('stopwords')

def main():
    location = ''
    if (len(sys.argv) > 1):
        if (sys.argv[1] in config.JOB_LOCATIONS):
            location = sys.argv[1]
        else:
            sys.exit('*** Please try again. *** \nEither leave it blank or input a city from this list:\n{}'.format('\n'.join(config.JOB_LOCATIONS)))
    jobs_info = web_scrapper.get_jobs_info(location)
    skill_match = skill_keyword_match(jobs_info)
    skill_match.extract_jobs_keywords()
    resume_skills = skill_match.extract_resume_keywords(config.SAMPLE_RESUME_PDF)
    top_job_matches = skill_match.cal_similarity(resume_skills.index, location)
    top_job_matches.to_csv(config.RECOMMENDED_JOBS_FILE+location+'.csv', index=False)
    print('File of recommended jobs saved')

if __name__ == "__main__": 
    main()

