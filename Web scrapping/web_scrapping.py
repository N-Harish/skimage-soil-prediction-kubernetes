'''
@author Dereck Jos and Harish Natarajan
Web Scrapping Project
'''
from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

bing_crawler = BingImageCrawler(downloader_threads=4,
                                storage={'root_dir': 'red_soil'})

bing_crawler.crawl(keyword='red soil', filters=None, offset=0, max_num=1000)
