# -*- coding: utf-8 -*-
import random, json
import numpy as np
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time, os
import config

max_results_per_city = 500

page_record_limit = 50
num_pages = int(max_results_per_city/page_record_limit)

def get_jobs_info(search_location):
    exists = os.path.isfile(config.JOBS_INFO_JSON_FILE)
    if exists:
        with open(config.JOBS_INFO_JSON_FILE, 'r') as fp:
            jobs_info = json.load(fp)            
    else:
        jobs_info = web_scrape(search_location)
    return jobs_info
        
def web_scrape(search_location):
    job_links = []
    start = time.time()
    driver = webdriver.Chrome(config.WEBDRIVER_PATH)
    job_locations = config.JOB_LOCATIONS
    if (len(search_location) > 0):
        job_locations = [search_location]

    for location in job_locations: 
        url = 'https://www.indeed.ca/jobs?q='+ config.JOB_SEARCH_WORDS + '&l=' \
        + location + '&limit=' + str(page_record_limit) + '&fromage='+ str(config.DAY_RANGE)
        driver.set_page_load_timeout(15)
        webdriver.DesiredCapabilities.CHROME["unexpectedAlertBehaviour"] = "accept"
        driver.get(url)
        time.sleep(3)  
        for i in range(num_pages):            
            try:
                for job_each in driver.find_elements_by_xpath('//*[@data-tn-element="jobTitle"]'):
                    job_link = job_each.get_attribute('href')
                    job_links.append({'location':location, 'job_link':job_link})                
                print ('scraping {} page {}'.format(location, i+1))
                driver.find_element_by_link_text('Next »').click()
            except NoSuchElementException:
                print ("{} finished".format(location))
                break        
            time.sleep(3)
    
    with open(config.JOBS_LINKS_JSON_FILE, 'w') as fp:
        json.dump(job_links, fp)
        
    jobs_info = []
    for job_lk in job_links:
        m = random.randint(1,5)
        time.sleep(m) 
        
        link = job_lk['job_link'] 
        driver.get(link)   
        
        location = job_lk['location']
       
        title = driver.find_element_by_xpath('//*[@class="icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title"]').text
        
        company = driver.find_element_by_xpath('//*[@class="icl-u-lg-mr--sm icl-u-xs-mr--xs"]').text
       
        if (len(driver.find_elements_by_xpath('//*[@class="jobsearch-JobMetadataHeader-item "]'))==0):
            salary = np.nan 
        else:
            salary = driver.find_element_by_xpath('//*[@class="jobsearch-JobMetadataHeader-item "]').text
        desc = driver.find_element_by_xpath('//*[@class="jobsearch-JobComponent-description icl-u-xs-mt--md"]').text
        jobs_info.append({'link':link, 'location':location, 'title':title, 'company':company, 'salary':salary, 'desc':desc})
    with open(config.JOBS_INFO_JSON_FILE, 'w') as fp:
        json.dump(jobs_info, fp)
    driver.quit()    
    end = time.time() 
    scaping_time = (end-start)/60.
    print('Took {0:.2f} minutes scraping {1:d} data scientist/engineer/analyst jobs'.format(scaping_time, len(jobs_info)))
    return jobs_info
