'''
    Standard Modules
'''

import urllib.request

'''
    External Modules
'''

import cx_Oracle
import pandas as pd

from tqdm import tqdm

username = 'WEBDATA'
password = 'WrCEcP2ZSKv7MDn4_U201905'
host = 'upd-ora-sby.hq.bc'
service_name = 'UPDDBOCI'

download_url = 'http://esb-tibco.hq.bc:25055/photoDb/getPhoto?photoId='

connection_string = '{}/{}@{}:{}/{}'.format(username,password,host,'1521',service_name)
connection = cx_Oracle.connect(connection_string)
query_cursor = connection.cursor()

query_data = '''
SELECT photo_1, idate 
FROM webdata.mobile_photoverif_logs t 
ORDER  BY t.idate DESC
'''

query_cursor.execute(query_data.strip())
data = pd.DataFrame(query_cursor, columns = ['photo_1','idate'])

data = data.dropna(subset = ['photo_1'])
data['valid_status'] = data['photo_1'].apply(lambda x : len(str(x)) > 10)
data = data[data['valid_status'] == True].reset_index()

data.to_csv('mobile_photoverif_logs.csv',index = False)

for i in tqdm(range(len(data))):
    try:
        urllib.request.urlretrieve(download_url + data['photo_1'][i], "Data/UPDDB/{}_{}.jpg".format(data['photo_1'][i], data['idate'][i]))
    except:
        pass 
        # print(data['photo_1'][i])


connection.close()