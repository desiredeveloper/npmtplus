import os
import re
import requests 
from bs4 import BeautifulSoup

req = requests.get('https://spoken-tutorial.org/')
soup = BeautifulSoup(req.content, 'html.parser')

course_select = soup.find(id='id_search_foss')
course_list = [course['value'] for course in course_select.find_all('option')][1:]

# lang_select = soup.find(id='id_search_language')
# lang_list = [lang['value'] for lang in lang_select.find_all('option')][1:]
lang_list = ["Bengali","Hindi"]

for lang in lang_list:
    for course in course_list:
            
        url = 'https://spoken-tutorial.org/tutorial-search/?search_foss={}&search_language={}'.format(course,lang)
        req = requests.get(url)
        soup = BeautifulSoup(req.content, 'html.parser')
        if soup.find(class_='no-record') is not None:
            continue
        
        lecture_list = [lecture.text for lecture in soup.select("div.title a")]

        os.makedirs('data/'+course,exist_ok=True)        
        for lecture in lecture_list:
            file1 = open("data/{}/{}.{}".format(course,lecture,lang), "w+")
            
            url = 'https://spoken-tutorial.org/watch/{}/{}/{}/'.format(course,lecture,lang)
            req = requests.get(url)
            soup = BeautifulSoup(req.content, 'html.parser')
            
            for resources in soup.select("h4.list-group-item-heading"):
                if resources.text==' Script':
                    transcript_url = resources.parent['href']
                
            req = requests.get(transcript_url)
            soup = BeautifulSoup(req.content, 'html.parser')
            
            for row in soup.select('tr')[1:]:
                curr_row = row.find_all('td')[1].text
                curr_row = re.sub('[ \t\n]+', ' ', curr_row)
                file1.write(curr_row)
                file1.write('\n')
    file1.close()