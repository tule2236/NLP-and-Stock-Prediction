# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class SamsungItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    Date = scrapy.Field()
    Title = scrapy.Field()
    Body = scrapy.Field() 
