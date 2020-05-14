# This file contains alot of the functions used in the starbucks capstone notebook
# just to keep the notebook cleaner and better change tracking
from tqdm.notebook import tqdm
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt
import seaborn as sns

def append_one_person_offer(person_offer_df, this_offer, person_id, offer_index, transcript_grouped,this_person):
    '''
    A function to generete a new df with person and offer per row,
    In many cases this will only add one row,.
    However ther's also alot of rows with multiple from 
    the same combination (user/offer), which may even overlap in time.
    
    Arguments:
    person_offer_df -- append to this
    this offer -- current offer
    person_id -- offer_index
    transcript_grouped -- only one offer type, one person
    this_person -- Information about current user
    
    Returns:
    person_offer_df -- as input but with the new row(s).
    '''

    to_be_appended = defaultdict(list)
    def append_final(item):
        # this metod takes complete item and appends it
        to_be_appended['person'].append(person_id)
        to_be_appended['offer'].append(offer_index)
        to_be_appended['start'].append(item['start'])
        to_be_appended['viewed_time'].append(item['view'])
        to_be_appended['completed_time'].append(item['complete'])
        to_be_appended['viewed'].append(item['view'] is not None)
        to_be_appended['completed'].append(item['complete'] is not None)
        to_be_appended['viewed_after'].append(False)

    current = []
    view_unknown = []
    validity = this_offer['duration']*24 #in hours
    debuglist=[]
    for row in transcript_grouped.itertuples():
        current_time = row.time # in hours
        current_event = row.event
        debuglist.append((row.event,row.time))
        if current and current[0]['start']+validity < current_time:
            append_final(current.pop(0))
        if len(current) == 1 and view_unknown:
            current[0]['view'] = view_unknown.pop(0)
        if current_event == 'offer received':
            current.append({'start':current_time,'view':None, 'complete':None})

        elif current_event=='offer viewed':
            if len(current)>1:
                view_unknown.append(current_time)
            elif current and view_unknown: # and :
                current[0]['view'] = view_unknown.pop(0)
                view_unknown.append(current_time)
            elif current:
                current[0]['view'] = current_time
            else:
                to_be_appended['viewed_after'][-1]=True

        elif current_event=='offer completed':
            if current[0]['view'] is None and view_unknown:
                current[0]['view'] = view_unknown.pop(0)
            current[0]['complete'] = current_time
            append_final(current.pop(0))
        else:
            raise Exception("Unknown event type")
    for item in current: # if there's anything left at the end
        if item['view'] is None and view_unknown:
            item['view'] = view_unknown.pop(0)
        append_final(item)
    
    # Add common cells for all these
    
    nrows = len(to_be_appended['viewed'])
    
    for x in ['gender','age','became_member_on','income']:
        to_be_appended[x] = [this_person[x]]*nrows
    for x in ['reward','difficulty','duration','offer_type',
              'id','email', 'mobile','social','web']:
        to_be_appended[x] = [this_offer[x]]*nrows

    new_df = pd.DataFrame(to_be_appended)
    if person_offer_df is not None:
        return person_offer_df.append(new_df, ignore_index = True) 
    else:
        return pd.DataFrame(new_df)
    
def create_person_offer(transcript,portfolio,profile):
    """ A function to generete a new df with person and offer per row,

    Arguments:
        transcript -- Dataframe that contains all events
        portfolio -- Dataframe that contains datails of offers 
        profile -- Dataframe that contains details about customers

    Returns:
        person_offer_df -- new DataFrame
    """    
    person_offer_df = None
    # This will not include transaction, so we need another new table for those.
    for (person_id, offer_index), transcript_grouped in tqdm(transcript.dropna(subset=['offer_index']).groupby(['person','offer_index'])):
        this_offer = portfolio.loc[offer_index]
        this_person = profile.loc[person_id]
        person_offer_df = append_one_person_offer(person_offer_df, this_offer, person_id, offer_index, transcript_grouped, this_person)
    return person_offer_df



def get_before_after_mean(x, person_transaction):
    """ Appends the mean of all transactions 
    before and after an event.
    Note, the Dataframe need to contain the
    columns to be filled (for faster excecution)

    Arguments:
        x -- Pandas series to be modified
        person_transaction  -- Dataframe that contains all transaction

    Returns:
        x  -- input with new columns
    """

    person_id = x.person
    
    def split_before_after(time, columns):
        """This function will split a dataframe
        into 3 parts, before, on and after

        Arguments:
            time -- time in hours
            columns -- series with current persons transaction

        Returns:
            before -- mean daily spending before event
            current -- mean daily spending on event day
            after -- mean daily spending after event
        """        
        if np.isnan(time):
            return [], [], []
        
        currentday = int(time)//24
        before = columns[:currentday]
        current = columns[currentday]
        after = columns[currentday+1:]
        return (before, current, after)
    
    def split_between(time_view,time_complete, duration, columns):
        if np.isnan(time_view):
            return []
        start = int(time_view)//24
        if np.isnan(time_complete):
            end = start + duration
        else:
            end = int(time_complete)//24
        return columns[start:end+1]
            
    def mean_weighted(items, reverse=False):
        '''Returns the mean, if empty return 0
        It gives  weighted version, the second is worth
        0.75 of the first, third 0.75 of second etc etc
        This will make sure that items bought near the complete
        date is more importand than a week or so later
        '''
        if reverse:
            items = reversed(items)
        f,div,result=1,0,0
        for x in items:
            result+=(x*f)
            div+=f
            f*=0.75
        if div:
            result/=div
        return result
    
    def mean0(items):
        '''Returns the mean, if empty return 0'''
        if len(items) > 0:
            return items.mean()
        else:
            return 0
    
    #when is an offer?
    # recived time is actually pretty useless, other than calculating the end time
    # so, an offers time, where we coumt the revenue, is all the days from viewed,
    # until completed, either by complete, or by expired
    # if we then take the mean from these days, 
    # and after and before as weighted, that should be rather fair, no?
    try:
        person_row = person_transaction.loc[person_id]
        col = person_transaction.columns
        before_start, current_start, after_start = split_before_after(x.start, col)
        before_view, current_view, after_view = split_before_after(x.viewed_time, col)
        before_complete, current_complete, after_complete = split_before_after(x.completed_time, col)

        x.before_start = mean0(person_row[before_start])
        x.same_day_start = person_row[current_start].sum()
        x.after_start = mean0(person_row[after_start])
    
        x.before_view = mean0(person_row[before_view])
        x.same_day_view = person_row[current_view].sum()
        x.after_view = mean0(person_row[after_view])
        
        x.before_complete = mean0(person_row[before_complete])
        x.same_day_complete = person_row[current_complete].sum()
        x.after_complete = mean0(person_row[after_complete])
        
        between = split_between(x.viewed_time,x.completed_time, x.duration, col)
        x.w_before = mean_weighted(person_row[before_view],reverse=True)
        x.sum_during = person_row[between].sum()
        x.mean_during = mean0(person_row[between])
        x.w_after = mean_weighted(person_row[after_complete])
    except KeyError as e:
        pass
        
    return x

def clean_cut_name(names):
    '''Takes a default pd.cut name and makes it
    more readable and without special characters.
    '''
    ret = []
    for s in names:
        lower = int(float(s[s.find('(')+1:s.find(',')]))
        upper = int(float(s[s.find(',')+1:s.find(']')]))
        ret.append(f"{s[:s.find('(')]}{lower}_to_{upper}")
    return ret

def range_to_cat(df,col,n):
    """takes a column of continious data
    and turn it into categorical data
    with a nice name for each
    {name}_{firstvalue}-{lastvalue}

    Arguments:
        df -- ipnut DataFrama
        col -- Column to be modified
        n -- number of categories

    Returns:
        df -- input dataframe with n new columns
        newcols -- list of name of the new columns
    """    
    cat_df, bins = pd.cut(df[col], n, retbins=True)
    newcols = pd.get_dummies(cat_df, prefix=col, prefix_sep='_', drop_first=True, dummy_na=False) 
    newcols.columns = clean_cut_name(newcols.columns)
    df = pd.concat([df, newcols], axis=1)
    return df, list(newcols)

def prepare_data(df,y_cols, x_cols, x_convert_to_cat=[]):
    assert len(set(x_cols).intersection(set(y_cols))) == 0
    for col in x_convert_to_cat:
        df, new_x = range_to_cat(df, *col)
        x_cols += new_x
        
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=y_cols)
    y = df[y_cols]
    
    X = df[x_cols].fillna(-1)

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    return X_train, X_test, y_train, y_test, x_cols

def train_model(model, X_train, y_train, X_test,y_test):
    '''
    Arguments:
    df -- dataframe to learn from
    model -- model used to train
    y_cols -- columns used for ground truth 
    x_cols -- columns to learn from
    x_conver_to cat -- list of tuples pairs, will be added to x
                    these should NOT be in x_cols too 
            
    Returns:
    '''

    #
    #fit the model and obtain pred response
    model.fit(X_train, y_train)
    y_test_preds = model.predict(X_test)
    y_train_preds = model.predict(X_train)
    y_test_proba = model.predict_proba(X_test)
    y_train_proba = model.predict_proba(X_train)


    return model, y_test_preds, y_train_preds, y_test_proba, y_train_proba


def print_classification_report(y_train,y_test,y_train_preds,y_test_preds):
    """Print a summary of the predicted result

    Arguments:
        y_train {[type]} -- [description]
        y_test {[type]} -- [description]
        y_train_preds {[type]} -- [description]
        y_test_preds {[type]} -- [description]
    """    
    try:
        for i in range(y_train.shape[-1]):
            test = (y_test.iloc[:,i].values, y_test_preds[:,i])
            train = (y_train.iloc[:,i].values, y_train_preds[:,i])
            print(f"---------------{y_train.columns[i]}------train:-------------")
            print(classification_report(*train))
            print(f"----TEST---")
            print(classification_report(*test))
    except Exception as e:
        try:
            print(f"--------train:-------------")
            print(classification_report(y_train, y_train_preds))
            print(f"---------TEST--------------")
            print(classification_report(y_test, y_test_preds))
        except Exception as e2:
            print('could not do report',e, e2)
            
def printRocCurves(y_test,y_test_proba,y_cols, x_axis_max=1, print_thold=False):

    #print(np.array(y_test).T.shape, np.array(y_test_proba).shape)
    #print(np.array(y_test).T.shape, np.array(y_test_proba)[...,1,None].shape)
    plt.figure(figsize=(5,5))
    if len( np.array(y_test_proba).shape) == 2:
        y_test_proba= np.array(y_test_proba)[...,1,None].T
    else:
        y_test_proba= np.array(y_test_proba)[:,:,1]
        
    for name, y, p in zip(y_cols, np.array(y_test).T, y_test_proba):
        fpr, tpr, thresholds= roc_curve(y,p)
        thresholds = np.clip(thresholds, 0.0, 1.0)
        roc_auc = auc(fpr, tpr)

        #Calculate thresholds
        bigt=0
        for ff,tt in zip(fpr,thresholds):
            if ff >= x_axis_max:
                bigt=tt
                break
        if bigt==1.0:
            bigt=0
        plt.plot(fpr, tpr, label=f'{name} - (a = {roc_auc:0.5f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, x_axis_max])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right",fancybox=True, framealpha=1)
    
    if print_thold:
        #this woks less well with multiple classes. for now, print last only
        ax2 = plt.gca().twinx()
        ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r', alpha = 0.3)
        ax2.set_ylabel(f'Threshold - {y_cols[-1]}',color='r')
        ax2.ticklabel_format(useOffset=False, style='plain')
        ax2.set_ylim([bigt,1.0])

    plt.show()

def plot_cmatrix(y_test, y_test_preds, y_cols):
    if len(y_test.shape)>1 and y_test.shape[1]>1:
        cm = confusion_matrix(y_test.values.argmax(axis=1), y_test_preds.argmax(axis=1))
    else:
        cm = confusion_matrix(y_test.values, y_test_preds)
        if len(y_cols)==1:
            y_cols = ["not " + y_cols[0],y_cols[0]]
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index = y_cols,
                    columns = y_cols)
    plt.figure(figsize = (5,5))
    sns.heatmap(df_cm, annot=True,fmt="d")
    plt.show()

def print_feature_importance(model, x_cols):
    # Features importances
    try:
        importances = model.feature_importances_
        # calculate standard deviation
        try:
            std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
        except:
            std=None
        indices = np.argsort(importances)[::-1]
        feature_list = x_cols
        ff = np.array(feature_list)

        # plot the figure
        plt.figure(figsize = (15,5))
        plt.title("Feature importances")
        if std is not None:
            yerr = std[indices] 
        else:
            yerr = None
        plt.bar(range(len(x_cols)), importances[indices],yerr=yerr)
        plt.xticks(range(len(x_cols)), ff[indices], rotation=90)
        plt.show()
    except Exception as e:
        print('could not print features importance',e)
