import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.special import logsumexp

# 1) Import the Court Elo functions (exactly as-is).
#    Make sure Court_ELO_IRL.py is in your Python path or same folder,
#    and that it exposes generate_and_find_similar_players(...).
from Court_ELO_IRL import generate_and_find_similar_players

##############################################################################
# IRL CODE (unchanged except for minor helper to load multiple CSVs)
##############################################################################

def label_server_by_point(df):
    df = df.sort_values(by=['shot_count']).copy()

    point_id = 0
    rows = []
    df_iter = df.to_dict('records')

    def set_server_receiver_points(row, server_name):
        p1_pts = row['ply1_points']
        p2_pts = row['ply2_points']
        if row['ply1_name'] == server_name:
            return p1_pts, p2_pts
        else:
            return p2_pts, p1_pts

    current_server = None
    is_new_point = True

    for i, row in enumerate(df_iter):
        shot_ct = row['shot_count']
        final_out = row['final_outcome']

        # If shot_count=1 or first row => new point
        if shot_ct == 1 or i == 0 or is_new_point:
            point_id += 1
            current_server = row['ply1_name']
            is_new_point = False

        row['point_id'] = point_id
        row['server_name'] = current_server

        sp, rp = set_server_receiver_points(row, current_server)
        row['server_points'] = sp
        row['receiver_points'] = rp

        rows.append(row)

        # If final_outcome=0 or 1 => point ended => next row => new point
        if final_out in [0, 1] and i < (len(df_iter)-1):
            is_new_point = True

    new_df = pd.DataFrame(rows)
    return new_df

def is_break_point(row):
    sp = row['server_points']
    rp = row['receiver_points']
    if sp == 99 or rp == 99:
        return False
    if (rp == 3 and sp <= 2) or (rp == 4 and sp == 3):
        return True
    return False

def who_won(row):
    if row['final_outcome'] == 1:
        return row['ply1_name']
    else:
        return row['ply2_name']

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    # Label each row with consistent server/receiver
    df = label_server_by_point(df)

    # Remove unknown outcomes if present
    df = df[df['final_outcome'] != 99].copy()

    df['winner_name'] = df.apply(who_won, axis=1)
    df['is_break_point'] = df.apply(is_break_point, axis=1)

    df_break = df[df['is_break_point']].copy()
    return df_break

def build_mdp(df_break):
    from collections import defaultdict

    def get_state(row):
        sc = row['shot_count'] if row['shot_count'] != 99 else 0
        sp = row['server_points'] if row['server_points'] != 99 else 0
        rp = row['receiver_points'] if row['receiver_points'] != 99 else 0
        return (sp, rp, sc)

    def get_action(row):
        desc = str(row['description'])
        if desc == '99':
            desc = '(NA)'
        return desc

    group_map = defaultdict(list)
    for idx, row in df_break.iterrows():
        s = get_state(row)
        a = get_action(row)
        group_map[(s, a)].append(idx)

    states, actions = set(), set()
    transitions = defaultdict(lambda: defaultdict(list))
    expert_sa = []

    for (s, a), row_idxs in group_map.items():
        states.add(s)
        actions.add(a)
        p_each = 1.0 / len(row_idxs)
        for ridx in row_idxs:
            # single-step => next state is None
            transitions[s][a].append((None, p_each, ridx))
        for _ in row_idxs:
            expert_sa.append((s, a))

    return list(states), list(actions), transitions, expert_sa

def feature_func(s, a):
    (sp, rp, sc) = s
    a_id = float(hash(a) % 10000)
    return np.array([1.0, sp, rp, sc, a_id])

def soft_value_iteration(states, actions, transitions, rewards, max_iter=100):
    V = {s: 0.0 for s in states}
    for _ in range(max_iter):
        newV = {}
        for s in states:
            log_candidates = []
            for a in actions:
                if a not in transitions[s]:
                    continue
                r_sa = rewards.get((s, a), 0.0)
                sum_exp_next = 0.0
                for (ns, p, ridx) in transitions[s][a]:
                    if ns is None:
                        sum_exp_next += p * 1.0
                    else:
                        sum_exp_next += p * np.exp(V[ns])
                if sum_exp_next < 1e-30:
                    sum_exp_next = 1e-30
                log_candidates.append(r_sa + np.log(sum_exp_next))
            newV[s] = logsumexp(log_candidates) if log_candidates else 0.0
        V = newV
    return V

def policy_prob(s, a, policy, transitions):
    if a not in transitions[s]:
        return 0.0
    r_sa = policy['rewards'].get((s, a), 0.0)
    sum_exp_next = 0.0
    for (ns, p, _) in transitions[s][a]:
        if ns is None:
            sum_exp_next += p * 1.0
        else:
            sum_exp_next += p * np.exp(policy['V'][ns])
    if sum_exp_next < 1e-30:
        sum_exp_next = 1e-30
    val = np.exp(r_sa + np.log(sum_exp_next) - policy['V'][s])
    return max(val, 0.0)

def compute_state_visitation_freq(states, actions, transitions, start_counts, policy, iters=50):
    freq = {s: 0.0 for s in states}
    for s, val in start_counts.items():
        freq[s] += val
    for _ in range(iters):
        newF = {ss: 0.0 for ss in states}
        for s in states:
            fs = freq[s]
            if fs < 1e-30:
                continue
            for a in actions:
                pa = policy_prob(s, a, policy, transitions)
                flow = fs * pa
                for (ns, p, _) in transitions[s][a]:
                    if ns is not None:
                        newF[ns] += flow * p
        freq = newF
    return freq

def maxent_irl(states, actions, transitions, expert_sa, epochs=100, lr=0.001):
    # Precompute phi for (s,a)
    phi_sa = {}
    for s in states:
        for a in actions:
            if a in transitions[s]:
                phi_sa[(s, a)] = feature_func(s, a)
    dim = len(next(iter(phi_sa.values())))

    # Expert feature counts
    expert_counts = np.zeros(dim)
    for (s, a) in expert_sa:
        expert_counts += phi_sa[(s, a)]

    # Start distribution from expert states
    start_counts = defaultdict(float)
    for (s, a) in expert_sa:
        start_counts[s] += 1
    tot = sum(start_counts.values())
    for s in start_counts:
        start_counts[s] /= tot

    # Initialize w
    w = np.zeros(dim)

    def get_reward(s, a):
        if (s, a) not in phi_sa:
            return -9999.0
        return np.dot(w, phi_sa[(s, a)])

    for _ in range(epochs):
        # Build rewards
        rewards = {}
        for (s, a) in phi_sa:
            rewards[(s, a)] = get_reward(s, a)
        # soft VI
        V = soft_value_iteration(states, actions, transitions, rewards, max_iter=100)
        policy = {'rewards': rewards, 'V': V}

        # state visitation freq
        freq = compute_state_visitation_freq(states, actions, transitions, start_counts, policy, iters=50)

        # model feature counts
        model_counts = np.zeros(dim)
        for s in states:
            fs = freq[s]
            if fs < 1e-30:
                continue
            for a in actions:
                if (s, a) not in phi_sa:
                    continue
                pa = policy_prob(s, a, policy, transitions)
                model_counts += fs * pa * phi_sa[(s, a)]

        grad = expert_counts - model_counts
        w += lr * grad
        w = np.clip(w, -20, 20)

    final_rewards = {}
    for (s, a) in phi_sa:
        final_rewards[(s, a)] = np.dot(w, phi_sa[(s, a)])
    final_V = soft_value_iteration(states, actions, transitions, final_rewards, max_iter=100)
    final_policy = {'rewards': final_rewards, 'V': final_V}
    return w, final_policy

def compute_probabilities(df_break, transitions, policy):
    df_break['state'] = list(
        zip(
            df_break['server_points'].replace(99, 0).astype(int),
            df_break['receiver_points'].replace(99, 0).astype(int),
            df_break['shot_count'].replace(99, 0).astype(int)
        )
    )

    row2winner = {}
    for idx, row in df_break.iterrows():
        row2winner[idx] = row['winner_name']

    def row_prob_server_win(row):
        s = row['state']
        if s not in transitions:
            return 0.0
        sum_server = 0.0

        def is_server(ridx):
            if ridx not in df_break.index:
                return False
            w_name = row2winner[ridx]
            return (w_name == row['server_name'])

        def pol_prob(s, a):
            if a not in transitions[s]:
                return 0.0
            r_sa = policy['rewards'].get((s, a), 0.0)
            sum_exp_next = 0.0
            for (ns, pp, rr) in transitions[s][a]:
                if ns is None:
                    sum_exp_next += pp * 1.0
                else:
                    sum_exp_next += pp * np.exp(policy['V'][ns])
            if sum_exp_next < 1e-30:
                sum_exp_next = 1e-30
            val = np.exp(r_sa + np.log(sum_exp_next) - policy['V'][s])
            return max(val, 0.0)

        for a in transitions[s]:
            p_a = pol_prob(s, a)
            sub = 0.0
            for (ns, pdup, ridx) in transitions[s][a]:
                if is_server(ridx):
                    sub += pdup
            sum_server += p_a * sub
        return sum_server

    from collections import defaultdict
    server_map = defaultdict(list)
    for idx, row in df_break.iterrows():
        p_s = row_prob_server_win(row)
        server_map[row['server_name']].append(p_s)

    results = {}
    for sname, probs in server_map.items():
        if len(probs) == 0:
            results[sname] = 0.0
        else:
            results[sname] = np.mean(probs)
    return results

# 2) HELPER FUNCTIONS TO LOAD MULTIPLE CSV AND FILTER BY OPPONENT

def all_csv_files_in_folder(folder):
    """
    Returns a list of all CSV files (full paths) in the given folder.
    """
    pattern = os.path.join(folder, '*.csv')
    return glob.glob(pattern)

def load_filtered_data(
    folder_csv,
    start_year=2010,
    end_year=2015,
    target_player='Novak_Djokovic',
    opponents_list=None
):
    """
    Loads ALL CSV from `folder_csv` whose filename suggests a match year in
    [start_year..end_year], but only retains rows where:
       - `target_player` is in the match (ply1_name or ply2_name),
       - the other player is in `opponents_list`.

    Returns a single DataFrame concatenating these filtered matches.
    """
    import os
    
    if opponents_list is None:
        opponents_list = []

    # Normalize underscores/spaces in the target player & opponents
    target_player_norm = target_player.replace('_', ' ').lower()
    opponents_list_norm = [p.replace('_', ' ').lower() for p in opponents_list]

    all_csv = all_csv_files_in_folder(folder_csv)  # same helper as before
    frames = []

    for csv_file in all_csv:
        # -------------------------------------
        # 1) PARSE YEAR FROM CSV FILENAME
        #    Example filename:
        #    "20131102-M-Paris_Masters-SF-Novak_Djokovic-Roger_Federer.csv"
        #    We'll take the first 4 chars => "2013"
        # -------------------------------------
        base_name = os.path.basename(csv_file)  # e.g. "20131102-M-Paris_Masters-SF-..."
        try:
            file_year = int(base_name[0:4])  # parse the first 4 digits as year
        except ValueError:
            # If the filename doesn't start with a valid year, skip it
            continue

        # If this file's year is outside [start_year..end_year], skip it
        if file_year < start_year or file_year > end_year:
            continue

        # -------------------------------------
        # 2) LOAD THE DATA
        # -------------------------------------
        df = pd.read_csv(csv_file)
        # Ensure the essential columns exist
        if 'ply1_name' not in df.columns or 'ply2_name' not in df.columns:
            continue

        # Convert underscores to spaces, lower-case
        df['ply1_name'] = df['ply1_name'].astype(str).str.replace('_', ' ').str.lower()
        df['ply2_name'] = df['ply2_name'].astype(str).str.replace('_', ' ').str.lower()

        # -------------------------------------
        # 3) FILTER FOR TARGET_PLAYER + OPPONENT
        # -------------------------------------
        mask_target = (df['ply1_name'] == target_player_norm) | (df['ply2_name'] == target_player_norm)
        if opponents_list_norm:
            mask_opp = (df['ply1_name'].isin(opponents_list_norm)) | (df['ply2_name'].isin(opponents_list_norm))
            df_filt = df[mask_target & mask_opp].copy()
        else:
            df_filt = df[mask_target].copy()

        if not df_filt.empty:
            frames.append(df_filt)

    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()  # empty if no matches found

# 3) MAIN: Putting it all together

def main():
    """
    Example usage:
      - We find all players within +/-100 Elo points of Roger Federer
        on Hard courts, from year 2008..2021, using Court_ELO's function.
      - Then we filter our tennisabstract-csv-v4 data to get only matches
        where Novak Djokovic faces any of those "similar Elo" players.
      - We run the IRL pipeline on that combined data.
      - This includes matches like 20131102-M-Paris_Masters-SF-Novak_Djokovic-Roger_Federer.csv
        if it exists in the CSV folder.
      - Finally, we compute break-point IRL results from those filtered matches.
    """

    # 1) Find similar-Elo players to Roger Federer on Hard courts
    df_similar = generate_and_find_similar_players(
        start_year=2010,
        end_year=2015,
        surface_filter='Hard',
        player_name='Roger Federer',  # "baseline" player for Elo
        elo_range=100,
        output_folder='../court_elo'
    )
    # Just get a list of player names
    similar_players = df_similar['Player'].tolist()
    print("\nPlayers within 100 Elo points of Roger Federer on Hard:")
    print(similar_players)

    # 2) Load matches where Novak Djokovic plays vs. those similar Elo players
    folder_csv = '../tennisabstract-csv-v4'
    df_all_filtered = load_filtered_data(
        folder_csv,
        target_player='Novak_Djokovic',
        opponents_list=similar_players
    )

    if df_all_filtered.empty:
        print("No matching data found for Novak Djokovic vs. Federer-similar group.")
        return

    print(f"Found {len(df_all_filtered)} total rows across matches of interest.")

    # 3) Among these rows, we specifically want break-point rows => run IRL
    df_all_filtered = label_server_by_point(df_all_filtered)
    df_all_filtered = df_all_filtered[df_all_filtered['final_outcome'] != 99].copy()
    df_all_filtered['winner_name'] = df_all_filtered.apply(who_won, axis=1)
    df_all_filtered['is_break_point'] = df_all_filtered.apply(is_break_point, axis=1)

    df_break = df_all_filtered[df_all_filtered['is_break_point']].copy()
    if df_break.empty:
        print("No break points found in the filtered dataset.")
        return

    # 4) Build the MDP from the break-point rows
    states, actions, transitions, expert_sa = build_mdp(df_break)

    # 5) MaxEnt IRL
    w, policy = maxent_irl(states, actions, transitions, expert_sa,
                           epochs=100, lr=0.001)

    # 6) Probability that the server wins break point
    server_win_map = compute_probabilities(df_break, transitions, policy)

    # 7) Print results
    print("\n=== Break-Point IRL Analysis (Djokovic vs Fed-similar set) ===")
    for server_name, pval in server_win_map.items():
        print(f"When {server_name} is serving a break point:")
        print(f"   Probability server wins: {pval:.3f}")
        print(f"   Probability receiver wins: {1.0 - pval:.3f}")

    print("\n=== Done ===")
    # print("- We used Court_ELO to find players similar to Roger Federer (within 100 Elo).")
    # print("- Filtered CSV rows to only Novak Djokovic vs those players.")
    # print("- Ran your IRL code for break-point situations on that subset of matches.")
    # print("- This includes analysis of the match 20131102-M-Paris_Masters-SF-Novak_Djokovic-Roger_Federer.csv,")
    # print("  if that file exists in your tennisabstract-csv-v4 folder.\n")


if __name__ == "__main__":
    main()
