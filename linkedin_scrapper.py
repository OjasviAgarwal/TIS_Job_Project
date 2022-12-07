import os
import logging
import config
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters, ExperienceLevelFilters, RemoteFilters

import pandas as pd

chrome_driver_path = os.path.join(os.path.dirname(__file__), "chromedriver")
jobs_data = []

logging.basicConfig(level = logging.INFO)

def on_data(data: EventData):
    print('[ON_DATA]', data.title, data.company, data.company_link, data.date, data.link, data.insights, len(data.description))
    jobs_data.append({'link': data.link, 'location': data.location, 'title': data.title, 'company': data.company, 'salary':'', 'desc': data.description})

def on_metrics(metrics: EventMetrics):
  print('[ON_METRICS]', str(metrics))

def on_error(error):
    print('[ON_ERROR]', error)

def on_end():
    print('[ON_END]')

scraper = LinkedinScraper(
    chrome_executable_path=chrome_driver_path,
    chrome_options=None,  
    headless=True,  
    max_workers=1,  
    slow_mo=1.5,
    page_load_timeout=20     
)

scraper.on(Events.DATA, on_data)
scraper.on(Events.ERROR, on_error)
scraper.on(Events.END, on_end)

queries = [
    Query(
        query='Software Developer',
        options=QueryOptions(
            locations=['United States', 'Canada'],            
            apply_link = True,  
            limit=5,
            filters=QueryFilters(              
                relevance=RelevanceFilters.RECENT,
                time=TimeFilters.MONTH,
                type=[TypeFilters.FULL_TIME, TypeFilters.INTERNSHIP],
                experience=None,                
            )
        )
    ),
]

def web_scrape():
    scraper.run(queries)
    df = pd.DataFrame(jobs_data, columns=['link', 'location', 'title', 'company', 'salary', 'desc'])
    df.to_csv("LinkedinJobs.csv", index=False)

def get_linkedin_jobs_info():
    exists = os.path.isfile(config.LINKED_JOBS_INFO_CSV_FILE)     
    if exists:
        df = pd.read_csv(config.LINKED_JOBS_INFO_CSV_FILE)
                   
    else:
        df = web_scrape()
    return df
