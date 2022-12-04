# Job Recommendation By Skill Match

Project to to create a simple skill-keyword-based job recommendation engine, which match keywords from resume to job descriptions

## Install

Install virtualenv
virtualenv is a tool to create isolated Python projects. Think of it, as a cleanroom, isolated from other virsions of Python and libriries.

Enter this command into terminal

sudo pip install virtualenv

or if you get an error

sudo -H pip install virtualenv

Start virtualenv
Navigate to where you want to store your code. Create new directory.

mkdir my_project && cd my_project

INSIDE my_project folder create a new virtualenv

virtualenv env

Activate virtualenv

source env/bin/activate

This project requires **Python 3.0+** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/) - pip install numpy
- [Pandas](http://pandas.pydata.org)
- [NLTK Stopwords](https://www.nltk.org/book/ch02.html)
- [Selenium](https://www.seleniumhq.org/)
- [PyPDF2](https://pythonhosted.org/PyPDF2/)
- pip install -r requirements.txt

## Code

Code is provided in 
- job_recommendation.py
- linkedin_scrapper.py
- skill_keyword_match.py
- web_scrapper.py

## Run

In a terminal or command window, navigate to the top-level project directory `TIS_Job_Project/` and run one of the following commands:

Search and match jobs in all cities:
```python indeed_job_recommendation.py```

Search and match jobs in one city e.g. Vancouver,BC:
```python indeed_job_recommendation.py Vancouver,BC```

When finishes successfully, it will say 'File of recommended jobs saved'.

## Data
Data collected from TBD 


