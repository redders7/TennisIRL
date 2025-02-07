import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.special import logsumexp
import shutil  # For file operations in PCSP functions

# Import the Court Elo functions (make sure Court_ELO_IRL.py is in your path)
from Court_ELO_IRL import generate_and_find_similar_players

##############################################################################
# Updated Feature Labels (20 features)
##############################################################################
FEATURE_LABELS = [
    "Bias",
    "Server Points",
    "Receiver Points",
    "Server Games",
    "Receiver Games",
    "Server Sets",
    "Receiver Sets",
    "Serve Type",
    "Rally Length",
    "UE Deuce Body",
    "UE Deuce T",
    "UE Deuce Wide",
    "UE Ad Body",
    "UE Ad T",
    "UE Ad Wide",
    "Previous Shot (Encoded)",
    "Shot Type (Encoded)",
    "Shot Placement (Encoded)",
    "Shot Depth (Encoded)",
    "Shot Outcome (Encoded)"
]

##############################################################################
# Label Points & Aggregate Unforced Error Counts
##############################################################################
def label_server_by_point(df):
    # Sort using shot_count to segment rallies
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
            
    def set_server_receiver_games(row, server_name):
        if row['ply1_name'] == server_name:
            return row['ply1_games'], row['ply2_games']
        else:
            return row['ply2_games'], row['ply1_games']
    
    def set_server_receiver_sets(row, server_name):
        if row['ply1_name'] == server_name:
            return row['ply1_sets'], row['ply2_sets']
        else:
            return row['ply2_sets'], row['ply1_sets']
    
    current_server = None
    is_new_point = True
    for i, row in enumerate(df_iter):
        shot_ct = row['shot_count']
        final_out = row['final_outcome']
        # Start a new point if shot_count == 1 (or at the beginning or after a point ends)
        if shot_ct == 1 or i == 0 or is_new_point:
            point_id += 1
            current_server = row['ply1_name']
            is_new_point = False
        row['point_id'] = point_id
        row['server_name'] = current_server
        sp, rp = set_server_receiver_points(row, current_server)
        row['server_points'] = sp
        row['receiver_points'] = rp
        sg, rg = set_server_receiver_games(row, current_server)
        row['server_games'] = sg
        row['receiver_games'] = rg
        ss, rs = set_server_receiver_sets(row, current_server)
        row['server_sets'] = ss
        row['receiver_sets'] = rs
        rows.append(row)
        # When final_outcome indicates point end, flag next row as new point.
        if final_out in [0, 1] and i < (len(df_iter) - 1):
            is_new_point = True
    new_df = pd.DataFrame(rows)
    # Compute rally length: number of shots in a rally (point)
    new_df['rally_length'] = new_df.groupby('point_id')['point_id'].transform('count')
    # Determine serve_type from the first shot in the point (typically a serve: shot_type==3)
    new_df['serve_type'] = new_df.groupby('point_id')['shot_type'].transform('first')
    
    # For unforced error (UE) counts, break them into six categories.
    def map_service_location(shot_placement):
        if shot_placement in [1,2,3]:
            if shot_placement == 1:
                return "deuce_t"
            elif shot_placement == 2:
                return "deuce_body"
            elif shot_placement == 3:
                return "deuce_wide"
        elif shot_placement in [4,5,6]:
            if shot_placement == 4:
                return "ad_t"
            elif shot_placement == 5:
                return "ad_body"
            elif shot_placement == 6:
                return "ad_wide"
        return None

    def compute_ue_counts(group):
        counts = {"deuce_body":0, "deuce_t":0, "deuce_wide":0, "ad_body":0, "ad_t":0, "ad_wide":0}
        for idx, r in group.iterrows():
            if r.get('shot_outcome') == 3:  # 3 indicates an unforced error
                loc = map_service_location(r.get('shot_placement'))
                if loc in counts:
                    counts[loc] += 1
        group['unforced_error_deuce_body'] = counts["deuce_body"]
        group['unforced_error_deuce_t'] = counts["deuce_t"]
        group['unforced_error_deuce_wide'] = counts["deuce_wide"]
        group['unforced_error_ad_body'] = counts["ad_body"]
        group['unforced_error_ad_t'] = counts["ad_t"]
        group['unforced_error_ad_wide'] = counts["ad_wide"]
        return group

    new_df = new_df.groupby('point_id').apply(compute_ue_counts)
    return new_df

##############################################################################
# Helper: Determine who won the point
##############################################################################
def who_won(row):
    if row['final_outcome'] == 1:
        return row['ply1_name']
    else:
        return row['ply2_name']

##############################################################################
# Data Loading (single CSV version)
##############################################################################
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df = label_server_by_point(df)
    df = df[df['final_outcome'] != 99].copy()
    df['winner_name'] = df.apply(who_won, axis=1)
    return df

##############################################################################
# Helper: List all CSV files in a folder
##############################################################################
def all_csv_files_in_folder(folder):
    pattern = os.path.join(folder, '*.csv')
    return glob.glob(pattern)

##############################################################################
# NEW: Load Filtered Data
##############################################################################
def load_filtered_data(folder_csv, start_year=2010, end_year=2015, target_player='Novak_Djokovic', opponents_list=None):
    if opponents_list is None:
        opponents_list = []
    target_player_norm = target_player.replace('_', ' ').lower()
    opponents_list_norm = [p.replace('_', ' ').lower() for p in opponents_list]
    all_csv = all_csv_files_in_folder(folder_csv)
    frames = []
    used_files = []
    for csv_file in all_csv:
        base_name = os.path.basename(csv_file)
        try:
            file_year = int(base_name[0:4])
        except ValueError:
            continue
        if file_year < start_year or file_year > end_year:
            continue
        df = pd.read_csv(csv_file)
        if 'ply1_name' not in df.columns or 'ply2_name' not in df.columns:
            continue
        df['ply1_name'] = df['ply1_name'].astype(str).str.replace('_', ' ').str.lower()
        df['ply2_name'] = df['ply2_name'].astype(str).str.replace('_', ' ').str.lower()
        mask_target = (df['ply1_name'] == target_player_norm) | (df['ply2_name'] == target_player_norm)
        if opponents_list_norm:
            mask_opp = (df['ply1_name'].isin(opponents_list_norm)) | (df['ply2_name'].isin(opponents_list_norm))
            df_filt = df[mask_target & mask_opp].copy()
        else:
            df_filt = df[mask_target].copy()
        if not df_filt.empty:
            frames.append(df_filt)
            used_files.append(csv_file)
    if frames:
        print("Files used in analysis:")
        for file in used_files:
            print(file)
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()

##############################################################################
# Build the MDP (with updated state & action definitions)
##############################################################################
def build_mdp(df):
    """
    State: (server_points, receiver_points, server_games, receiver_games, server_sets,
            receiver_sets, serve_type, rally_length, UE_vector, prev_shot_type)
      where UE_vector = (unforced_error_deuce_body, unforced_error_deuce_t, unforced_error_deuce_wide,
                         unforced_error_ad_body, unforced_error_ad_t, unforced_error_ad_wide)
    
    Action: (shot_type, shot_placement, shot_depth, shot_outcome)
    """
    def get_state(row):
        return (
            row.get('server_points', 0),
            row.get('receiver_points', 0),
            row.get('server_games', 0),
            row.get('receiver_games', 0),
            row.get('server_sets', 0),
            row.get('receiver_sets', 0),
            row.get('serve_type', 0),
            row.get('rally_length', 0),
            (
                row.get('unforced_error_deuce_body', 0),
                row.get('unforced_error_deuce_t', 0),
                row.get('unforced_error_deuce_wide', 0),
                row.get('unforced_error_ad_body', 0),
                row.get('unforced_error_ad_t', 0),
                row.get('unforced_error_ad_wide', 0)
            ),
            row.get('prev_shot_type', 0)
        )
    def get_action(row):
        return (
            row.get('shot_type', 0),
            row.get('shot_placement', 0),
            row.get('shot_depth', 0),
            row.get('shot_outcome', 0)
        )
    group_map = defaultdict(list)
    for idx, row in df.iterrows():
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
            transitions[s][a].append((None, p_each, ridx))
        for _ in row_idxs:
            expert_sa.append((s, a))
    return list(states), list(actions), transitions, expert_sa

##############################################################################
# Feature function (using the updated state and action)
##############################################################################
def feature_func(s, a):
    (sp, rp, sg, rg, ss, rs, serve_t, rl, ue, prev_shot) = s
    (shot_type, shot_placement, shot_depth, shot_outcome) = a

    prev_shot_enc = float(hash(prev_shot) % 1000)
    shot_type_enc = float(hash(shot_type) % 1000)
    shot_placement_enc = float(hash(shot_placement) % 1000)
    shot_depth_enc = float(hash(shot_depth) % 1000)
    shot_outcome_enc = float(hash(shot_outcome) % 1000)

    ue_values = list(ue) if isinstance(ue, (list, tuple)) else [ue]
    feats = np.array([1.0, sp, rp, sg, rg, ss, rs, serve_t, rl] + ue_values +
                     [prev_shot_enc, shot_type_enc, shot_placement_enc, shot_depth_enc, shot_outcome_enc])
    return feats

##############################################################################
# Soft Value Iteration, Policy, and MaxEnt IRL
##############################################################################
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
    
    # Initialize weights
    w = np.zeros(dim)
    
    def get_reward(s, a):
        if (s, a) not in phi_sa:
            return -9999.0
        return np.dot(w, phi_sa[(s, a)])
    
    for _ in range(epochs):
        rewards = {}
        for (s, a) in phi_sa:
            rewards[(s, a)] = get_reward(s, a)
        V = soft_value_iteration(states, actions, transitions, rewards, max_iter=100)
        policy = {'rewards': rewards, 'V': V}
        
        freq = compute_state_visitation_freq(states, actions, transitions, start_counts, policy, iters=50)
        
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
    
    print("\n=== Reward Weights ===")
    for label, weight in zip(FEATURE_LABELS, w):
        print(f"{label}: {weight:.4f}")
    
    return w, final_policy

##############################################################################
# NEW FUNCTIONS: Output IRL Learned Transition Probabilities & PCSP Integration
##############################################################################
def output_transition_probabilities(policy, transitions, output_file="irl_transition_probs.txt"):
    """
    For each state-action pair, compute the probability (using policy_prob) 
    and write a line with the state, action, and probability.
    """
    with open(output_file, "w") as f:
        for s in transitions:
            for a in transitions[s]:
                prob = policy_prob(s, a, policy, transitions)
                f.write(f"State: {s}, Action: {a}, Probability: {prob}\n")
    print(f"Learned transition probabilities saved to {output_file}")

# PCSP evaluation function (example; adjust paths as needed)
def eval_pcsp(pcsp_dir, file_name):
    os.system('cp %s ../PAT351/%s' % (os.path.join(pcsp_dir, file_name), file_name))
    os.chdir('../PAT351')
    os.system('mono PAT3.Console.exe -pcsp %s %s.txt' % (file_name, file_name[:-5]))
    with open('%s.txt' % file_name[:-5]) as f:
        lines = f.readlines()
    os.system('rm %s' % file_name)
    os.system('rm %s.txt' % file_name[:-5])
    os.chdir('../src')
    if len(lines) < 5 or '[' not in lines[3] or ']' not in lines[3]:
        print(lines)
        return -1, -1
    min_max_probs = [float(score) for score in lines[3].split('[')[1].split(']')[0].split(',')]
    return min_max_probs

def generate_pcsp_from_irl(irl_file, date, ply1_name, ply2_name, ply1_hand, ply2_hand, ply_dim=8):
    """
    Generate a PCSP file using the IRL learned transition probabilities file.
    (Here we simply copy the IRL file into a PCSP folder with an appropriate name.)
    """
    pcsp_dir = f'../pcsp_files_irl/{date[:4]}'
    if not os.path.isdir(pcsp_dir):
        os.mkdir(pcsp_dir)
    file_name = f"{ply1_name}_{ply2_name}_irl.pcsp"
    dest_path = os.path.join(pcsp_dir, file_name)
    shutil.copy(irl_file, dest_path)
    # Call the external PCSP evaluator (assumes the PAT351 project is set up as in your example)
    pcsp_result = eval_pcsp(pcsp_dir, file_name)
    return pcsp_result

##############################################################################
# MAIN: IRL Pipeline & PCSP Integration (Single Model)
##############################################################################
def main():
    """
    Pipeline:
      1. Use generate_and_find_similar_players to find players of similar Elo (on a specified court).
      2. Load matches where the target player (Novak Djokovic) faces one of these similar players.
      3. Process the data to label points and outcomes.
      4. Build the MDP based on the defined state–action pairs.
      5. Run the MaxEnt IRL model to learn reward weights and derive the policy.
      6. Output the learned transition probabilities for each state–action pair.
      7. Plug these values into the PCSP model and generate a PCSP file (to be run on PAT3).
    """
    # 1) Find similar opponents using a reference player (e.g., Roger Federer on Hard courts)
    df_similar = generate_and_find_similar_players(
        start_year=2010,
        end_year=2015,
        surface_filter='Hard',
        player_name='Roger Federer',
        elo_range=100,
        output_folder='../court_elo'
    )
    similar_players = df_similar['Player'].tolist()
    print("\nPlayers within 100 Elo points of Roger Federer on Hard:")
    print(similar_players)
    
    # 2) Load matches where the target player (Novak Djokovic) faces one of these similar players.
    folder_csv = '../tennisabstract-csv-v4'
    df_all_filtered = load_filtered_data(
        folder_csv,
        start_year=2010,
        end_year=2015,
        target_player='Novak_Djokovic',
        opponents_list=similar_players
    )
    
    if df_all_filtered.empty:
        print("No matching data found for Novak Djokovic vs. similar players.")
        return
    print(f"Found {len(df_all_filtered)} total rows of match data.")
    
    # 3) Process all points: label points and compute winners.
    df_all_filtered = label_server_by_point(df_all_filtered)
    df_all_filtered = df_all_filtered[df_all_filtered['final_outcome'] != 99].copy()
    df_all_filtered['winner_name'] = df_all_filtered.apply(who_won, axis=1)
    
    # 4) Build the MDP from the processed data.
    states, actions, transitions, expert_sa = build_mdp(df_all_filtered)
    
    # 5) Run the IRL model (MaxEnt IRL) to learn reward weights and derive the policy.
    w, policy = maxent_irl(states, actions, transitions, expert_sa, epochs=100, lr=0.001)
    
    # 6) Output learned transition probabilities for each state–action pair.
    output_file = "irl_transition_probs.txt"
    output_transition_probabilities(policy, transitions, output_file)
    
    # 7) Generate a PCSP file using the IRL transition probabilities and evaluate it using PAT3.
    date_str = "2025-02-07"  # Example date; adjust as needed.
    ply1_name = "Novak Djokovic"
    ply2_name = "Roger Federer"
    ply1_hand = "RH"        # Assume right-handed for the target player.
    ply2_hand = "RH"        # Placeholder.
    pcsp_result = generate_pcsp_from_irl(output_file, date_str, ply1_name, ply2_name, ply1_hand, ply2_hand, ply_dim=8)
    print("\n=== PCSP Simulation Result using IRL Transition Probabilities ===")
    print(pcsp_result)
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
