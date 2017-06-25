import gensim
import nltk
from nltk.corpus import stopwords
from joblib import Parallel, delayed
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models, utils
import numpy as np
from nltk.tokenize import sent_tokenize
import re
from textblob import TextBlob
import PIL
import seaborn as sns


#LOAD THE MODEL
model = gensim.models.ldamodel.LdaModel.load('lda_az_5topics.gz', mmap=None)

#CONNECT TO DATABASE
def connect_db(address,dbname,username):
    ## 'engine' is a connection to a database
    ## Here, we're using postgres, but sqlalchemy can connect to other things too.
    engine = create_engine('postgres://%s@%s/%s'%(username,address,dbname))
    print(engine.url)

    ## create a database (if it doesn't exist)
    if not database_exists(engine.url):
        print("database %s doesn't exist! creating..." % dbname)
        create_database(engine.url)
    
    con = None
    con = psycopg2.connect(database = dbname, user = username)
    return(con)


#get data from database
def get_data(con,business_name):
#     query = """
# SELECT r.*, b.name,b.stars as business_stars,b.latitude,b.longitude,
# b.city,b.state,b.postal_code, b.address FROM reviews r
# LEFT JOIN 
#  businesses b
#  ON b.business_id = r.business_id
#  WHERE b.name=%(bizname)s AND state='AZ';
# """  

    query = """
    SELECT * FROM prepared 
    WHERE name=%(bizname)s AND state='AZ';
    """

    bizdf = pd.read_sql(query,con,params= {'bizname': business_name})
    return(bizdf)


#NLP STUFF
stop_words = stopwords.words('english')

def preprocess(doc):
    doc = gensim.utils.simple_preprocess(doc)
    doc = [word for word in doc if word not in stop_words]
    return(doc)
    
def tabulate_topics(model,num_topics = -1, num_words=10,shape='long'):
    
    tops = model.show_topics(num_topics = num_topics,num_words=num_words,formatted=False)
    
    alldf = []

    for i,j in tops:
        tmp = pd.DataFrame(j,columns=['word','prob'])
        tmp.loc[:,'topic'] = np.repeat(i,len(tmp))
        alldf.append(tmp)

    tmp = pd.concat(alldf,axis=0)
    
    if shape=='wide':
        tmp = tmp.set_index('word')
        tmp = pd.pivot_table(tmp, values='prob', index=['word'],columns=['topic'], aggfunc=np.sum)
    return(tmp)
    

def raw2bow(docs,dictionary):
    preproc = [preprocess(d) for d in docs]
    bow = [dictionary.doc2bow(d,allow_update=False) for d in preproc]
    return(bow)


ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
NON_ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENGLISH_STOPWORDS
 
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}
 
def get_language(text):
    words = set(nltk.wordpunct_tokenize(text.lower()))
    return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key = lambda x: x[1])[0]
 
def is_english(text):
    text = text.lower()
    words = set(nltk.wordpunct_tokenize(text))
    return len(words & ENGLISH_STOPWORDS) > len(words & NON_ENGLISH_STOPWORDS)


def topics_by_review(model,df,top_only=False):
    
    bows = raw2bow(df.text,model.id2word)   
    allprobs = []
    
    for i,tops in zip(df.review_id,model.get_document_topics(bows)):
        probs = pd.DataFrame(tops,columns=['topic','prob'])
        probs.loc[:,'review_id'] = np.repeat(i,len(probs))
        allprobs.append(probs)
        
    rev_topics = pd.merge(df,pd.concat(allprobs),on='review_id')   
    
    if top_only:
        idx = rev_topics.groupby(['review_id'])['prob'].transform(max) == rev_topics['prob']
        rev_topics = rev_topics[idx].reset_index(drop=True)
    else:
        rev_topics['top_prob'] = rev_topics.groupby(['review_id'])['prob'].transform(max)

    return(rev_topics)


def sentence_topics(model,rawtext,top_only=False):
    
    sent_tokenize_list = sent_tokenize(rawtext)
    preproc = [preprocess(x) for x in sent_tokenize_list]
    bows = [model.id2word.doc2bow(x,allow_update=False) for x in preproc]
    tops = model.get_document_topics(bows)
    
    output = []
    
    for i,sent in enumerate(sent_tokenize_list):
        if len(preproc[i])> 5:
     
            topics = tops[i]
            topics_df = pd.DataFrame(np.vstack(topics),columns=['topic','prob'])
            sent_df = pd.DataFrame(np.repeat(sent,topics_df.shape[0]),columns=['sentence'])
                                     
            tmp = pd.concat([sent_df,topics_df],axis=1)
            output.append(tmp)
        
    if len(output)>0:    
        output = pd.concat(output).reset_index(drop=True)
    else:
        return(None)
    
    if top_only:
        idx = output.groupby(['sentence'])['prob'].transform(max) == output['prob']
        output = output[idx].reset_index(drop=True)
    else:
        output['top_prob'] = output.groupby(['sentence'])['prob'].transform(max)
        
    return(output)
    

def topics_by_sentence(model,df,top_only=False):
    
    output = []

    for i in range(len(df)):
        
        cleantext = re.sub(r'^https?:\/\/.*[\r\n]*', '', df.text[i], flags=re.MULTILINE)
        
        if len(cleantext)>0:
            sents = sentence_topics(model,cleantext,top_only=top_only) 
            
            if isinstance(sents,pd.DataFrame):
                sents['review_id'] = np.repeat(df.review_id[i],len(sents))
                output.append(sents)
    
    if len(output)>0:
        output = pd.concat(output)
        output['sentiment'] = [TextBlob(x).sentiment.polarity for x in output.sentence]
    else:
        return(None)
    

    return(output)


    

def calc_ratings(model,df,top_only = False):
    df = df.copy()
    df.loc[:,'sentiment'] =[TextBlob(x).sentiment.polarity for x in df.text]
    rev_topics = topics_by_review(model,df,top_only=top_only)
    sents = rev_topics.groupby('topic',as_index=False)[['sentiment','stars']].agg(np.mean)
    return(sents)


def get_example_sentences(model,df,topics = None, topn = 3,sentiment_scores = None):
    
    if topics is None: 
        topics = np.arange(model.num_topics)
    
    output = topics_by_sentence(model,df,top_only=True)
    
    if isinstance(output,pd.DataFrame):
        output = output[output['topic'].isin(topics)].reset_index(drop=True)
        grouped = output.groupby('topic')['prob'].nlargest(topn)
        sentences = output.loc[grouped.index.levels[1].values,'sentence'] 

        topsents = pd.DataFrame(sentences).reset_index(drop=True)
        topsents['topic'] = grouped.reset_index()['topic']

    else:
        #generate an empty data frame to return
        topsents = pd.DataFrame(np.arange(5),columns=['topic'])
        topsents['sentence'] = np.repeat([''],5)
        
    return(topsents)
    
def make_wordclouds(model,save=True,fname='topic',num_words=20):

    if isinstance(model,pd.DataFrame):
        topics = model
    else:
        topics = tabulate_topics(model,num_words=num_words)
    
    tops = topics.groupby('topic')

    for  i,group in tops:
        text = ' '.join(group.word.values)

        # Generate a word cloud image
        wordcloud = WordCloud(background_color=None,mode='RGBA',relative_scaling=.5,colormap='viridis').generate(text)

        # Display the generated image:
        # the matplotlib way:
        plt.figure(num=None, figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
    #     plt.title('Topic %d' % i)

        if save: 
            plt.savefig('%s_%02d.png' %(fname,i),transparent=False,bbox_inches='tight',pad_inches=0)




def make_barplots(sents,save=False,fname='figure.png'):

    fig, ax1 = plt.subplots(figsize=(10,2.5))
    
    yvals = np.arange(0,sents.shape[0]*2,2)
    

    for i in range(len(sents)):
        ax1.text(2.25, yvals[i]+.45, 'Star Rating')
    ax1.barh(yvals,sents.stars,linewidth=.5,color='gold')
    ax1.barh(yvals,np.repeat(5,len(sents)),fill=False,edgecolor='black',linewidth=.5)

    ax1.set_xlim([0,5.03])
    ax1.xaxis.set_visible(False) 
    ax1.yaxis.set_visible(False) 
    ax1.axis('off')

    ax2 = ax1.twiny()
    yvals = np.arange(1,sents.shape[0]*2,2)
      
    
    for i in range(len(sents)):
        ax2.text(-.1, yvals[i]+.45, 'Sentiment')
    ax2.barh(yvals, sents.sentiment,color='green',linewidth=.5)
    ax2.barh(yvals,np.repeat(-1,len(sents)),color='white',fill=False,edgecolor='black',linewidth=.5)
    ax2.barh(yvals,np.repeat(1,len(sents)),color='white',fill=False,edgecolor='black',linewidth=.5)
    
    ax2.set_xlim([-1,1.01])
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.axis('off')
    
    if save:
        fig.savefig(fname,transparent=False,bbox_inches='tight',pad_inches=0)
        
    return(fig)


def make_topic_plot(data,save=False,fname='topic_dist.png'):

    sns.set(font_scale=3) 
    sns.set_style("white")

    fig = plt.figure(figsize=(12,10))
    ax = sns.barplot(y='topic',x='val',data=data,orient='h',palette='viridis')
    sns.despine()
#     ax.axes.get_xaxis().set_visible(False)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.set_ylabel('')    
    ax.set_xlabel('')
    
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

    plt.savefig(fname,transparent=False,bbox_inches='tight',pad_inches=0)

def combine_images(img1,img2,outfile=None,direction='vertical'):

    image1 = PIL.Image.open(img1)

    image2 = PIL.Image.open(img2)

    if direction=='vertical':
        image2.thumbnail((image1.width,image1.height),PIL.Image.ANTIALIAS)
        blank = PIL.Image.new('RGB', (image2.width,image1.height+image2.height))
        blank.paste(image2, (0,image1.height))
    else:
        image2.thumbnail((image1.width,image1.height),PIL.Image.ANTIALIAS)
        blank = PIL.Image.new('RGB', (image1.width + image2.width,image1.height))
        blank.paste(image2, (image1.width,0))

    
    blank.paste(image1, (0,0))
        
    if outfile is not None:
        blank.save(outfile)
    
    return(blank)

def get_business_topics(topics,reviews,topn = 30):
    
    tops = topics.topic.unique()
    
    alldf = []
    
    for topnum in tops:
    
        topic_subset = topics[topics.topic==topnum]
        rev_subset = reviews[reviews.topic==topnum]
        
        if len(rev_subset)> 0:
            revwords = preprocess(rev_subset.text.values.tolist()[0])


            rev = set(revwords)
            topic = set(topic_subset.word.values.tolist())

            overlap = rev & topic

            overlap_idx = topic_subset.word.isin(overlap)

            tmp = topic_subset.loc[overlap_idx,['word','prob']].sort_values(by='prob',ascending=False).nlargest(topn,'prob')
            tmp['topic'] = np.repeat(topnum,len(tmp))
            alldf.append(tmp)
        
    
    alldf = pd.concat(alldf)
        
    return(alldf)

