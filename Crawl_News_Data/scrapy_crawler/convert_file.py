import csv
import pandas as pd
lst = []
a = ['VZ', 'WMT', 'WFC']

for each in a:
	name = each + '_News.csv'
	with open(name) as csvfile:
		reader = csv.reader(csvfile)
		# writer = csv.writer(writefile,delimiter="|")
		for row in reader:
			date = row[1][55:63]
			month = str(date[:2]).zfill(2)
			day = str(date[2:4])
			year = date[4:]
			end = str(month+'/'+day+'/'+year)
			lst.append([end,row[2],row[0]])
	df = pd.DataFrame(lst)	
	
	df.to_csv(name, header=False, index=False)
	print(df.head())	

