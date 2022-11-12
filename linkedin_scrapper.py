import os
import logging
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters, ExperienceLevelFilters, RemoteFilters

import pandas as pd

chrome_driver_path = os.path.join(os.path.dirname(__file__), "chromedriver")
jobs_data = []

# Change root logger level (default is WARN)
logging.basicConfig(level = logging.INFO)



# Fired once for each successfully processed job
def on_data(data: EventData):
    print('[ON_DATA]', data.title, data.company, data.company_link, data.date, data.link, data.insights, len(data.description))
    jobs_data.append({'link': data.link, 'location': data.location, 'title': data.title, 'company': data.company, 'salary':'', 'desc': data.description})
# Fired once for each page (25 jobs)
def on_metrics(metrics: EventMetrics):
  print('[ON_METRICS]', str(metrics))

def on_error(error):
    print('[ON_ERROR]', error)

def on_end():
    print('[ON_END]')


scraper = LinkedinScraper(
    chrome_executable_path=chrome_driver_path, # Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver) 
    chrome_options=None,  # Custom Chrome options here
    headless=True,  # Overrides headless mode only if chrome_options is None
    max_workers=1,  # How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
    slow_mo=1.5,  # Slow down the scraper to avoid 'Too many requests 429' errors (in seconds)
    page_load_timeout=20  # Page load timeout (in seconds)    
)

# Add event listeners
scraper.on(Events.DATA, on_data)
scraper.on(Events.ERROR, on_error)
scraper.on(Events.END, on_end)

queries = [
    Query(
        query='Software Developer',
        options=QueryOptions(
            locations=['United States', 'Canada'],            
            apply_link = True,  # Try to extract apply link (easy applies are skipped). Default to False.
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

scraper.run(queries)
df = pd.DataFrame(jobs_data, columns=['link', 'location', 'title', 'company', 'salary', 'desc'])
df.to_csv("LinkedinJobs.csv", index=False)