
a = ['IBM.N','T.N', 'CVX.N', 'VZ.N', 'WMT.N', 'WFC.N']
initial = 'http://www.reuters.com/finance/stocks/companyNews?symbol=WFC.N&date=01012017/'
myfile = open('url_list.txt','w')
for year in list(range(2011,2018)):
	year_link = initial[:-5] + str(year) + initial[-1:]
	for month in list(range(1,13)):
		month_link = year_link[:-9] + str(month).zfill(2) + year_link[-7:]
		for day in list(range(1,32)):
			day_link = month_link[:-7]+ str(day).zfill(2) +month_link[-5:]
			# print(day_link)
			myfile.write(day_link+'\n')
myfile.close()
