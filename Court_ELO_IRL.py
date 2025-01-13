# File: Court_ELO.py

import datetime
import numpy as np
import os
import pandas as pd
import ELO_538 as elo
import tqdm
import re
import glob

# Constants for surface types
SURFACES = ['Hard', 'Clay', 'Grass']

RET_STRINGS = (
    'ABN', 'DEF', 'In Progress', 'RET', 'W/O', ' RET', ' W/O', 'nan', 'walkover'
)
ABD_STRINGS = (
    'abandoned', 'ABN', 'ABD', 'DEF', 'def', 'unfinished', 'Walkover'
)

# Normalize player names
def normalize_name(s, tour='atp'):
    if tour=='atp':
        s = s.replace('-',' ')
        s = s.replace('Stanislas','Stan').replace('Stan','Stanislas')
        s = s.replace('Alexandre','Alexander')
        s = s.replace('Federico Delbonis','Federico Del').replace('Federico Del','Federico Delbonis')
        s = s.replace('Mello','Melo')
        s = s.replace('Cedric','Cedrik')
        s = s.replace('Bernakis','Berankis')
        s = s.replace('Hansescu','Hanescu')
        s = s.replace('Teimuraz','Teymuraz')
        s = s.replace('Vikor','Viktor')
        s = s.rstrip()
        s = s.replace('Alex Jr.','Alex Bogomolov')
        s = s.title()
        sep = s.split(' ')
        return ' '.join(sep[:2]) if len(sep)>2 else s
    else:
        return s

# Format match dataframe
def format_match_df(df, tour, ret_strings=[], abd_strings=[]):
    cols = ['tourney_id','tourney_name','surface','draw_size','tourney_date','match_num',
            'winner_name','loser_name','score','best_of','w_svpt','w_1stWon','w_2ndWon',
            'l_svpt','l_1stWon','l_2ndWon']
    df = df[cols]
    df = df.rename(columns={'winner_name':'w_name','loser_name':'l_name',
                            'tourney_id':'tny_id','tourney_name':'tny_name','tourney_date':'tny_date'})

    df['w_name'] = [normalize_name(x,tour) for x in df['w_name']]
    df['l_name'] = [normalize_name(x,tour) for x in df['l_name']]
    df['tny_name'] = ['Davis Cup' if 'Davis Cup' in s else s for s in df['tny_name']]
    df['tny_name'] = [s.replace('Australian Chps.','Australian Open').replace('Australian Open-2','Australian Open')
                        .replace('U.S. National Chps.','US Open') for s in df['tny_name']]
    df['is_gs'] = (df['tny_name']=='Australian Open') | (df['tny_name']=='Roland Garros') \
                 | (df['tny_name']=='Wimbledon')       | (df['tny_name']=='US Open')

    # format dates
    df['tny_date'] = [datetime.datetime.strptime(str(x), "%Y%m%d").date() for x in df['tny_date']]
    df['match_year'] = [x.year for x in df['tny_date']]
    df['match_month'] = [x.month for x in df['tny_date']]
    # correct december start dates
    df['match_year'] = df['match_year'] + (df['match_month'] == 12)
    df['match_month'] = [1 if month==12 else month for month in df['match_month']]
    df['score'] = [re.sub(r"[\(\[].*?[\)\]]", "", str(s)) for s in df['score']]
    df['score'] = ['RET' if 'RET' in s else s for s in df['score']]
    df['w_swon'] = df['w_1stWon']+df['w_2ndWon']
    df['l_swon'] = df['l_1stWon']+df['l_2ndWon']
    df['w_rwon'] = df['l_svpt'] - df['l_swon']
    df['l_rwon'] = df['w_svpt'] - df['w_swon']
    df['w_rpt']  = df['l_svpt']
    df['l_rpt']  = df['w_svpt']
    df.drop(['w_1stWon','w_2ndWon','l_1stWon','l_2ndWon'], axis=1, inplace=True)

    # remove matches involving retirement or abandonment
    abd_d, ret_d = set(abd_strings), set(ret_strings)
    df['score'] = ['ABN' if score.split(' ')[-1] in abd_d else score for score in df['score']]
    df['score'] = ['RET' if score in ret_d else score for score in df['score']]
    df = df.loc[(df['score']!='ABN') & (df['score']!='RET')].reset_index(drop=True)
    return df

# Load and process match data across multiple years
def concat_data(start_y, end_y, tour='atp'):
    match_year_list = []
    for i in range(start_y, end_y+1):
        f_name = f'../match_data/{tour}_matches_{i}.csv'
        try:
            df = pd.read_csv(f_name)
        except:
            print(f'Could not read file {f_name}')
            continue
        format_df = format_match_df(df, tour, RET_STRINGS, ABD_STRINGS)
        out_name = f'../match_data_formatted/{tour}_matches_{i}.csv'
        format_df.to_csv(out_name, index=False)
        match_year_list.append(format_df)
    if not match_year_list:
        return pd.DataFrame()
    full_match_df = pd.concat(match_year_list, ignore_index=True)
    full_match_df = full_match_df.sort_values(by=['tny_date','tny_name','match_num'], ascending=True)
    return full_match_df.reset_index(drop=True)

def generate_court_specific_elo_dict(arr, counts_538):
    """
    arr => Nx4 array or N rows DataFrame e.g. [winner_name, loser_name, surface, some_score_col?]
    We do the Elo update. For demonstration, we do the basic approach from ELO_538.
    """
    SURFACES = ['Hard','Clay','Grass']
    elo_dict = {surf:{} for surf in SURFACES}

    for i in range(len(arr)):
        w_name, l_name, surface, score = arr[i][:4]
        if surface not in elo_dict: 
            continue
        # create rating if not exist
        if w_name not in elo_dict[surface]:
            elo_dict[surface][w_name] = elo.Rating()
        if l_name not in elo_dict[surface]:
            elo_dict[surface][l_name] = elo.Rating()

        rater = elo.Elo_Rater()
        w_rating = elo_dict[surface][w_name]
        l_rating = elo_dict[surface][l_name]
        # The "score" param or "counts_538" might define some margin or weighting. We'll do basic approach:
        new_w, new_l = rater.rate_1vs1(w_rating, l_rating, 1.0, counts_538)  # winner=1.0
        elo_dict[surface][w_name] = new_w
        elo_dict[surface][l_name] = new_l

    return elo_dict

def generate_and_find_similar_players(
    start_year, end_year, surface_filter, 
    player_name, elo_range=100, 
    tour='atp', 
    output_folder='../court_elo'
):
    """
    1) Loads matches in [start_year..end_year].
    2) Filters by surface
    3) Builds Elo dict on that surface
    4) Gets player's Elo => finds players within 'elo_range' Elo
    5) Saves to CSV => also returns the DF so IRL code can read directly
    """
    full_data = concat_data(start_year, end_year, tour)
    if full_data.empty:
        print("No data loaded.")
        return pd.DataFrame()

    surface_df = full_data[ full_data['surface']==surface_filter ]
    if surface_df.empty:
        print(f"No matches found on surface={surface_filter}")
        return pd.DataFrame()

    # Build array => [w_name, l_name, surface, score]
    matches_arr = surface_df[['w_name','l_name','surface','score']].values
    elo_dict = generate_court_specific_elo_dict(matches_arr, counts_538=None)

    # Retrieve target player's Elo on this surface
    if player_name not in elo_dict[surface_filter]:
        print(f"Player {player_name} not found in Elo dictionary on {surface_filter}.")
        return pd.DataFrame()
    player_elo = elo_dict[surface_filter][player_name].value

    # find who is within 'elo_range'
    similar_map = {}
    for p, rating in elo_dict[surface_filter].items():
        if abs(rating.value - player_elo) <= elo_range:
            similar_map[p] = rating.value

    df_similar = pd.DataFrame(list(similar_map.items()), columns=['Player','Elo'])
    df_similar = df_similar.sort_values(by='Elo', ascending=False)

    # save
    os.makedirs(output_folder, exist_ok=True)
    out_csv = f'{output_folder}/court-elo-{elo_range}_{player_name}_{surface_filter}_{start_year}_{end_year}.csv'
    df_similar.to_csv(out_csv, index=False)
    print(f"Similar Elo players saved to {out_csv}")

    return df_similar


# Example usage / test
if __name__=="__main__":
    # Example: we find players with Elo within 100 points of 'Roger Federer' on Grass in 2015-2016
    df_output = generate_and_find_similar_players(
        start_year=2015, end_year=2016, 
        surface_filter='Grass', 
        player_name='Roger Federer', 
        elo_range=100,
        tour='atp',
        output_folder='../court_elo'
    )
    print(df_output.head(10))
