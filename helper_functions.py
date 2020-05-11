# This file contains alot of the functions used in the starbucks capstone notebook
# just to keep the notebook cleaner and better change tracking
from tqdm.notebook import tqdm
import pandas as pd
from collections import defaultdict
import numpy as np

def append_one_person_offer(person_offer_df, this_offer, person_id, offer_index, transcript_grouped,this_person):
    '''
    A function to generete a new df with person and offer per row,
    In many cases this will only add one row,.
    However ther's also alot of rows with multiple from 
    the same combination (user/offer), which may even overlap in time.
    
    INPUT:
    person_offer_df: append to this
    this offer: current offer
    person_id: offer_index
    transcript_grouped: only one offer type, one person
    this_person: Information about current user
    
    OUTPUT:
    person_offer_df: as input but with the new row(s).
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

        
    #todo, doublecheck all times
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
    '''
    A function to generete a new df with person and offer per row,

    INPUT:
    '''
    person_offer_df = None
    # This will not include transaction, so we need anoyher new table for those.
    for (person_id, offer_index), transcript_grouped in tqdm(transcript.dropna(subset=['offer_index']).groupby(['person','offer_index'])):
        this_offer = portfolio.loc[offer_index]
        this_person = profile.loc[person_id]
        person_offer_df = append_one_person_offer(person_offer_df, this_offer, person_id, offer_index, transcript_grouped, this_person)
    return person_offer_df



def get_before_after_mean(x, person_transaction):

    person_id = x.person
    
    def split_before_after(time, columns):
        if np.isnan(time):
            before = []
            current = []
            after = []
        else:
            currentday = int(time)//24
            before = columns[:currentday]
            current = columns[currentday]
            after = columns[currentday+1:]
        return (before, current, after)
    
    def mean0(items):
        if len(items) > 0:
            return items.mean()
        else:
            return 0
    
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
    except KeyError as e:
        pass
        
    return x

