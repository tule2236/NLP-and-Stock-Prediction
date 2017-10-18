import pandas as pd 
import os
import time
from datetime import datetime

# path = "Users:/tule/Downloads/intraQuarter"

# def Key_Stats(gather = "Total Debt/Equity (mrq)"):
# 	print('a')
# 	statspath = '_KeyStats/'
# 	print(statspath)
# 	print(os.walk(statspath))
# 	stock_list = [x[0] for x in os.walk(statspath)]
# 	# print(stock_list)

# 	for each_dir in stock_list[1:]:
# 		print('b')
# 		each_file = os.listdir(each_dir)
# 		if len(each_file) > 0:
# 			for file in each_file:
# 				date_stamp = datetime.strptime(file, '%Y%m%d%H%M%S.html')
# 				unix_time = time.mktime(date_stamp.timetuple())
# 				print(date_stamp, unix_time)
# 				time.sleep(15)

# Key_Stats()
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))