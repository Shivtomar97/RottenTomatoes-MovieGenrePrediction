#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:31:18 2018

@author: jaynanda
"""

animation_link = open('/Users/jaynanda/Desktop/Assignments/660/Project/Movie Links/animation.txt').readlines()

from bs4 import BeautifulSoup
import re
import requests


def run(url):
  
    pageLink = url
    for i in range(0,5):
      try:
          response=requests.get(pageLink,headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', } )        
          html=response.content
          break
      except Exception as e:# browser.open() threw an exception, the attempt to get the response failed
          None
    soup = BeautifulSoup(html.decode('ascii', 'ignore'),'html.parser') # parse the html
    #for i in range(0,5):
    try:
        description = soup.find('div', {'class':re.compile('movie_synopsis')}).text
    except:
        description = 'NA'
    #for i in range(0,5):
    try:
        box = soup.findAll('div', {'class':re.compile('meta-value')})
        rating = box[0].text
      
    except:
        rating = 'NA'   
    #for i in range(0,5):
    try:
        box4 = soup.findAll('div', {'class':re.compile('meta-value')})
        director_1 = box4[2].findAll(('a', {'href':re.compile('celebrity')}))
        director1 = director_1[0].text
      
    except:
        director1 =  'NA'    
    #for i in range(0,5):
    try:
        box5 = soup.findAll('div', {'class':re.compile('meta-value')})
        director_2 = box5[2].findAll(('a', {'href':re.compile('celebrity')}))
        director2 = director_2[1].text
      
    except:
        director2 = 'NA' 
    #for i in range(0,5):
    try:
        box6 = soup.findAll('div', {'class':re.compile('meta-value')})
        director_3 = box6[2].findAll(('a', {'href':re.compile('celebrity')}))
        director3 = director_3[2].text
      
    except:
        director3 = 'NA' 
    #for i in range(0,5):
    try:
        box7 = soup.findAll('div', {'class':re.compile('meta-value')})
        writer_1 = box7[3].findAll(('a', {'href':re.compile('celebrity')}))
        writer1 = writer_1[0].text
      
    except:
        writer1 = 'NA'    
    #for i in range(0,5):         
    try:
        box8 = soup.findAll('div', {'class':re.compile('meta-value')})
        writer_2 = box8[3].findAll(('a', {'href':re.compile('celebrity')}))
        writer2 = writer_2[1].text
      
    except:
        writer2 = 'NA'   
    #for i in range(0,5): 
    try:
        box9 = soup.findAll('div', {'class':re.compile('meta-value')})
        writer_3 = box9[3].findAll(('a', {'href':re.compile('celebrity')}))
        writer3 = writer_3[2].text
      
    except:
        writer3 =  'NA'    
    #for i in range(0,5):
    try:
        box10 = soup.findAll('div', {'class':re.compile('meta-value')})
        writer_4 = box10[3].findAll(('a', {'href':re.compile('celebrity')}))

        writer4 = writer_4[3].text
      
    except:
        writer4 = 'NA' 
    #for i in range(0,5):
    try:
        time1 = soup.findAll('time')
        releasedate = time1[0].text
      
    except:
        releasedate =  'NA'    
    #for i in range(0,5):
    try:
        time2 = soup.findAll('time')

        streamdate = time2[1].text
      
    except:
        streamdate = 'NA'  
    #for i in range(0,5):
    try:
        time3 = soup.findAll('time')
        runtime = time3[2].text
      
    except:
        runtime = 'NA'
    #for i in range(0,5):    
    try:
        box11 = soup.findAll('div', {'class':re.compile('meta-value')})
        studio =  box11[-1].text
      
    except:
        studio = 'NA' 
    #for i in range(0,5):      
    try:
        cast_1 = soup.findAll('div',{'class':re.compile('cast-item')})
        cast1 =  cast_1[0].find('span').text
      
    except:
        cast1 = 'NA'  
    #for i in range(0,5):
    try:
        cast_2 = soup.findAll('div',{'class':re.compile('cast-item')})
        cast2 =  cast_2[1].find('span').text
      
    except:
        cast2 = 'NA' 
    #for i in range(0,5):      
    try:
        cast_3 = soup.findAll('div',{'class':re.compile('cast-item')})
        cast3 =  cast_3[2].find('span').text
      
    except:
        cast3 = 'NA'  
    #for i in range(0,5):  
    try:
        cast_4= soup.findAll('div',{'class':re.compile('cast-item')})
        cast4 =  cast_4[3].find('span').text
      
    except:
        cast4 = 'NA'
    #for i in range(0,5):      
    try:
        cast_5 = soup.findAll('div',{'class':re.compile('cast-item')})
        cast5 =  cast_5[4].find('span').text
     
    except:
        cast5 = 'NA'
    #for i in range(0,5):      
    try:
        cast_6 = soup.findAll('div',{'class':re.compile('cast-item')})
        cast6 =  cast_6[5].find('span').text
      
    except:
        cast6 = 'NA'
    #for i in range(0,5):      
    try:
        title = soup.find('h1',{'class':re.compile('title')}).text
        year = soup.find('span',{'class':re.compile('year')}).text
      
    except:
        title = 'NA'  
        year = 'NA'
    return (description,rating,director1,director2,director3,writer1,writer2,writer3,writer4,releasedate,streamdate,runtime,studio,cast1,cast2,cast3,cast4,cast5,cast6,title,year) 

animation = []
for i in range(0,len(animation_link)):
    animation.append(run(animation_link[i])) 
    print(i)
   
for i in range(0,25):
    c = []
    for i in range(0,len(animation)):
        if animation[i][0] == 'NA':
            c.append(i)
            
    for item in c:
        animation[item] = (run(animation_link[item]))

        
import pandas as pd
animation = pd.DataFrame(animation)

import os
path= r'/Users/jaynanda/Desktop/Assignments/660/Project/Unprocessed Data'
animation.to_csv(os.path.join(path, r'animation.csv'), index=False)