from flask import render_template, url_for
from flask import request
from demo import app
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
import pandas as pd
from flask import request
import psycopg2
from project_lib import *
# from textrank import extract_key_phrases, extract_sentences
import json
import time
from scipy.stats import zscore


print('kind of working')

#connect to database
user = 'insightdb' #add your username here (same as previous postgreSQL)                      
host = 'mydbinstance.cgaia5rww7ar.us-west-2.rds.amazonaws.com'
dbname = 'project'

con=psycopg2.connect(dbname= dbname, host=host, port= 5432, user= user, password= 'datascience')

@app.route('/')
@app.route('/index')
@app.route('/input')
def demo_input():
    query = """
SELECT  distinct(name) FROM businesses
 WHERE  state='AZ' AND review_count>=200 AND categories LIKE '%Restaurants%'
 ORDER BY name;
"""
    query_results = pd.read_sql_query(query,con)
    # print(query_results)

    return render_template("index.html", businesses = query_results.name)


@app.route('/dashboard')
def fancy():
  biz_name = request.args.get('biz_name')
  print(biz_name)

  #get all data from that business
  rtops= get_data(con,biz_name)

  rtops['sentiment'] =[TextBlob(x).sentiment.polarity for x in rtops.text]

  topnames = ['topic_%02d' % x for x in range(5)]
  topnames_w = ['topic_%02d_w' % x for x in range(5)]

  for t in topnames:
      rtops[t+'_w'] =rtops.loc[:,t]*rtops.sentiment

  rtops['score'] = rtops[topnames_w].sum(axis=1) 
  total_score = rtops.score.mean()   

  #score each review based on topic probability, usefulness flag, and recency
  timedeltas = pd.to_datetime(time.strftime("%Y-%m-%d")) - pd.to_datetime(rtops.review_date)
  r = timedeltas.rank(ascending=False)

  rtops['sent_score'] = rtops.top_topic_prob * (rtops.useful) * r
  tmp = rtops.groupby(['top_topic']).apply(lambda x: x.nlargest(3,'sent_score')).reset_index(drop=True) #grab the top 3 for each topic
  

  #grab example sentences for each one
  examples = get_example_sentences(model,tmp,topn=10)
  
  #get topic distribution (based on dominant topic)
  tdist = pd.crosstab(rtops.name,rtops.top_topic)
  tdist= tdist.stack()
  tdist = tdist.groupby(['name']).transform(lambda x: x/np.sum(x))
  
  #get the mean sentiment and stars within each top_topic
  sdist = rtops.groupby(['name','top_topic'])['stars','sentiment'].mean()


  #merge
  alldat = pd.concat([tdist,sdist],axis=1,join='outer')
  alldat = alldat.reset_index()
  alldat.columns = ['name','topic','prob','stars','sentiment']

  t1 = set(range(5))
  t2 = set(alldat.topic)
  print(t1,t2)

  
  missing = np.array(list(t1.difference(t2)))
  
  print(missing)

  if  len(missing)>0:
    missing2 = np.zeros([len(missing),3])
    missing = np.hstack([missing[:,np.newaxis],missing2])
    print(missing)
    alldat = alldat.append(pd.DataFrame(missing, columns = ['topic','prob','stars','sentiment']))
    alldat = alldat.sort_values('topic')



  topic_names = np.array(['Ambience', 'Service', 'Food', 'Speed', 'International'])

  alldat['topic'] = topic_names[alldat.topic.values.astype(int)]

  
  # rtops['review_date'] = pd.DatetimeIndex(rtops.review_date)
  # dat2 = rtops.copy()
  # dat2.set_index('review_date', inplace=True)
  # dat3= dat2.resample('7D')[['sentiment','stars']].mean().ffill().rolling(52).mean()
  # data_over_time = dat3.dropna().agg(zscore)


  # data_over_time.plot()
  # plt.savefig('./demo/static/time.png', dpi=199, transparent=False,bbox_inches='tight',pad_inches=0)

  return render_template('dashboard.html', root_path = app.root_path, examples=examples, scores = json.dumps(alldat.to_dict(orient='records')), biz_name = biz_name)


