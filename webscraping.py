
from bs4 import BeautifulSoup


import requests


url1 = 'https://sunnah.com/bukhari/8'
url2 = 'https://sunnah.com/bukhari/9'
url3 = 'https://sunnah.com/bukhari/10'
url4 = 'https://sunnah.com/bukhari/11'
url5 = 'https://sunnah.com/bukhari/12'
url6 = 'https://sunnah.com/bukhari/14'
url7 = 'https://sunnah.com/bukhari/18'
url8 = 'https://sunnah.com/bukhari/19'
url9 = 'https://sunnah.com/bukhari/20'
url10 = 'https://sunnah.com/bukhari/22'


page1 = requests.get(url1)
scrape1 = BeautifulSoup(page1.content,"lxml")

page2 = requests.get(url2)
scrape2 = BeautifulSoup(page2.content,"lxml")

page3 = requests.get(url3)
scrape3 = BeautifulSoup(page3.content,"lxml")

page4 = requests.get(url4)
scrape4 = BeautifulSoup(page4.content,"lxml")

page5 = requests.get(url5)
scrape5 = BeautifulSoup(page5.content,"lxml")

page6 = requests.get(url6)
scrape6 = BeautifulSoup(page6.content,"lxml")

page7 = requests.get(url7)
scrape7 = BeautifulSoup(page7.content,"lxml")

page8 = requests.get(url8)
scrape8 = BeautifulSoup(page8.content,"lxml")

page9 = requests.get(url9)
scrape9 = BeautifulSoup(page9.content,"lxml")

page10 = requests.get(url10)
scrape10 = BeautifulSoup(page10.content,"lxml")

with open("./data/solat/shalat.txt", 'w') as f:
	for content1 in scrape1.find_all('div', attrs = {'class': 'arabic_hadith_full arabic'}):

		print(content1.text)
	
		f.write("%s\n" % content1.text)

with open("./data/solat/shalat1.txt", 'w') as f:
	for content2 in scrape2.find_all('div', attrs = {'class': 'arabic_hadith_full arabic'}):

		print(content2.text)
	
		f.write("%s\n" % content2.text)

with open("./data/solat/shalat2.txt", 'w') as f:
	for content3 in scrape3.find_all('div', attrs = {'class': 'arabic_hadith_full arabic'}):

		print(content3.text)
	
		f.write("%s\n" % content3.text)


with open("./data/solat/shalat3.txt", 'w') as f:
	for content4 in scrape4.find_all('div', attrs = {'class': 'arabic_hadith_full arabic'}):

		print(content4.text)
	
		f.write("%s\n" % content4.text)

with open("./data/solat/shalat4.txt", 'w') as f:
	for content5 in scrape5.find_all('div', attrs = {'class': 'arabic_hadith_full arabic'}):

		print(content5.text)
	
		f.write("%s\n" % content5.text)

with open("./data/solat/shalat5.txt", 'w') as f:
	for content6 in scrape6.find_all('div', attrs = {'class': 'arabic_hadith_full arabic'}):

		print(content6.text)
	
		f.write("%s\n" % content6.text)

with open("./data/solat/shalat6.txt", 'w') as f:
	for content7 in scrape7.find_all('div', attrs = {'class': 'arabic_hadith_full arabic'}):

		print(content7.text)
	
		f.write("%s\n" % content7.text)

with open("./data/solat/shalat7.txt", 'w') as f:
	for content8 in scrape8.find_all('div', attrs = {'class': 'arabic_hadith_full arabic'}):

		print(content8.text)
	
		f.write("%s\n" % content8.text)

with open("./data/solat/shalat8.txt", 'w') as f:
	for content9 in scrape9.find_all('div', attrs = {'class': 'arabic_hadith_full arabic'}):

		print(content9.text)
	
		f.write("%s\n" % content9.text)

with open("./data/solat/shalat9.txt", 'w') as f:
	for content10 in scrape10.find_all('div', attrs = {'class': 'arabic_hadith_full arabic'}):

		print(content10.text)
	
		f.write("%s\n" % content10.text)

