# Scrape News Data to Train Text Classifier
I decided to use Scrapy because this tools work great on crawling dynamic websites, especially with those having a well-structured URL links that we can formulized with jQuery. In this project, I choose to scrape news data from http://www.reuters.com because the financial news are categoried into separate company tags which are perfect for our project purspose. Moreover, the URL links are well structured in the format

` https://www.reuters.com/finance/stocks/company-news/ + Company Symbol + Date` 

I decided not to scrape the full article but only the summary diplayed on the main pages since it gives good overview of the full content. Moreover, since we will process a big news dataset of 22 companies in the 6-year time span from 2011 to 2017, I try to make each data record as compact as possible to speed up the processing time. The code of this project can be found in my GitHub account

## Installation
Create a virtual environment
```
pip install virtualenv
mkvirtualenv Crawl_News_Data
```
Install Scrapy and dependencies
```
pip install scrapy service_identity ipython pillow
```
Can check whether we have successfully installed Scrapy by typing command
```
pip freeze
```

## Development Setup
Create the first Spider
To create a Spider project, simply type
```
scrapy startproject Crawl_News_Data
``` 
This will create a project directory ***Crawl_News_Data*** and all the neccessary files for the crawler. Now I access the project directory

cd Crawl_News_Data
Display all available Scrapy command by

scrapy -h
Create the first Spider named samsung_Spider followed with the link of the news website that I want to scrape data from.
```
scrapy genspider samsung_Spider www.reuter.com
```

Up to this point, my directory will be like this:
```
samsung
    spiders
        __init__.py
        samsung_Spider.py
    __init__.py
    url_list.txt
    link_generator.py
    convert_file.py
    items.py
    middlewares.py
    pipelines.py
    settings.py
```

## Running the Test
I write a script `link_generator.py` to generate all the URL links of Reuters news of 22 companies from January 1, 2011 to December 31, 2017. Each company would have 365 days x 6 years = 2190 links, and I end up with 2190 x 22 = 48180 links stored in `url_list.txt`
Then, I construct my Spider in `samsung_Spider.py`.

After specifying the crawling rule, I let the Spider crawl the website
```
scrapy crawl samsung_Spider
```
I want to write the scraped news related to each company to a corresponsing csv file, so I can combine the news data with the corresponding stock data for the later sentiment analysis
```
scrapy crawl --nolog --output=Samsung_news.csv samsung_Spider
```
The source code doesn't contain the date in a clear structure, so I obtained the date from the URL links and then, I write a script `convert_file.py` to extract the date from the URL and re-format them to MM/DD/YY
The scraped data is stored in folder `News`

