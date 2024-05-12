
import numpy as np
import pandas as pd
import os
import json
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
import matplotlib.pyplot as plt

def plot_logits_over_time():
    # Create a scatter plot of 'greenKey_prob' over timesteps, colored by 'greenKey', with lines connecting dots of the same episode
    plt.figure(figsize=(10, 6))
    for episode in df_mamba['episode'].unique():
        episode_data = df_mamba[df_mamba['episode'] == episode]
        plt.plot(episode_data['timestep'], episode_data['greenKey_prob'], '-o', alpha=0.6, label=f'Episode {episode}', c="blue" if episode_data['greenKey'].iloc[0] else "orange")
        plt.scatter(episode_data['timestep'], episode_data['greenKey_prob'], c=episode_data['greenKey'], cmap='viridis', alpha=0.6)
    plt.title('Visualization of greenKey_prob over Timesteps')
    plt.xlabel('Timestep')
    plt.ylabel('Probability of Green Key')
    plt.grid(True)
    plt.legend(title='Episode')
    plt.show()

def train_linear_probe(embeddings, labels, label_name, type='logistic'):
    # Create a logistic regression model for the linear probe
    if type == 'logistic':
        linear_probe = LogisticRegression(max_iter=1000, random_state=42)
    elif type == 'ridge':
        linear_probe = Ridge(alpha=1.0)
    else:
        raise ValueError(f"Invalid type: {type}. Must be 'logistic' or 'ridge'.")

    # Fit the model using the reshaped embeddings as features and direction as the target
    linear_probe.fit(embeddings, labels)
    score = linear_probe.score(embeddings, labels)
    print(f"Accuracy of {label_name} linear probe:", score)

    return linear_probe, score

def train_reverse_probe_ridge(y, X, y_name):
    """
    Train a reverse probe using Random Forest from y to X.
    
    Parameters:
    y (pd.Series): Target variable.
    X (np.ndarray): Input features.
    y_name (str): Name of the target variable for logging purposes.
    
    Returns:
    rf (RandomForestRegressor): Trained Random Forest model.
    """
    from sklearn.preprocessing import StandardScaler
    
    # Ensure y is in the correct shape (n_samples,)
    # y = y.ravel()
    
    # Standardize the features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    # Initialize and fit Random Forest

    # rf = LinearRegression()
    rf = Ridge(alpha=1.0)
    halfway = len(X_scaled)//2
    rf.fit(X_scaled, y_scaled)
    
    # Print the R^2 score of the model
    score = rf.score(X_scaled, y_scaled)
    print(f"R^2 score of the {y_name} model:", score)
    return rf, score

def parse_hidden_state_dim(embeddings_file):
    seq_number = int(embeddings_file.split("seq")[-1].split("_")[0])

    transformer_hidden_state_dim = seq_number * 64
    lstm_hidden_state_dim = 2 * 64
    mamba_hidden_state_dim = seq_number * 64

    if "transformer" in embeddings_file:
        hidden_state_dim = transformer_hidden_state_dim
    elif "lstm" in embeddings_file:
        hidden_state_dim = lstm_hidden_state_dim
    elif "mamba" in embeddings_file:
        hidden_state_dim = mamba_hidden_state_dim
    else:
        hidden_state_dim = 64

    return hidden_state_dim

def linear_probe_wrapper(embeddings_file):
    hidden_state_dim = parse_hidden_state_dim(embeddings_file)

    # read JSON
    with open(embeddings_file, 'r') as f:
        embeddings_json = json.load(f)

    df = pd.DataFrame(embeddings_json)
    embeddings = np.array(df['embedding'].tolist()).reshape(-1, hidden_state_dim)
    greenKey = pd.DataFrame(df['greenKey'])
    agent_pos_x = pd.DataFrame(df['agentPosX'])
    direction = pd.DataFrame(df['direction'])
    timestep = pd.DataFrame(df['timestep'])

    green_probe, green_score = train_linear_probe(embeddings, greenKey, 'greenKey', type='logistic')
    _, agent_pos_x_score = train_linear_probe(embeddings, agent_pos_x, 'agentPosX', type='ridge')
    _, direction_score = train_linear_probe(embeddings, direction, 'direction', type='logistic')
    _, timestep_score = train_linear_probe(embeddings, timestep, 'timestep', type='ridge')

    df['greenKey_prob'] = np.array(green_probe.predict_proba(embeddings))[:, 1]
    greenKey_prob = pd.DataFrame(df['greenKey_prob'])

    _, green_reverse_score = train_reverse_probe_ridge(embeddings, greenKey, 'greenKey')
    _, green_reverse_prob_score = train_reverse_probe_ridge(embeddings, greenKey_prob, 'greenKey_prob')
    _, agent_pos_x_reverse_score = train_reverse_probe_ridge(embeddings, agent_pos_x, 'agentPosX')
    _, direction_reverse_score = train_reverse_probe_ridge(embeddings, direction, 'direction')
    _, timestep_reverse_score = train_reverse_probe_ridge(embeddings, timestep, 'timestep')

    # Combine greenKey, agent_pos_x, and direction into a single DataFrame
    combined_features = pd.concat([greenKey, agent_pos_x, direction, timestep], axis=1)
    combined_features.columns = ['greenKey', 'agentPosX', 'direction', 'timestep']
    combined_features_prob = pd.concat([greenKey_prob, agent_pos_x, direction, timestep], axis=1)
    combined_features_prob.columns = ['greenKey_prob', 'agentPosX', 'direction', 'timestep']
    combined_features_without_key = pd.concat([agent_pos_x, direction, timestep], axis=1)
    combined_features_without_key.columns = ['agentPosX', 'direction', 'timestep']

    # Train a linear probe on the combined features
    combined_probe, combined_score = train_reverse_probe_ridge(embeddings, combined_features, 'combinedFeatures')
    combined_probe_prob, combined_prob_score = train_reverse_probe_ridge(embeddings, combined_features_prob, 'combinedFeatures_prob')
    combined_probe_without_key, combined_without_key_score = train_reverse_probe_ridge(embeddings, combined_features_without_key, 'combinedFeatures_without_key')

    avg_reward = np.mean(sorted(df['reward'], reverse=True)[:30])

    results = {
        "key": green_score,
        "agent_pos_x": agent_pos_x_score,
        "direction": direction_score,
        "timestep": timestep_score,
        "key_reverse": green_reverse_score,
        "key_prob_reverse": green_reverse_prob_score,
        "agent_pos_x_reverse": agent_pos_x_reverse_score,
        "direction_reverse": direction_reverse_score,
        "timestep_reverse": timestep_reverse_score,
        "combined_reverse": combined_score,
        "combined_prob_reverse": combined_prob_score,
        "combined_no_key_reverse": combined_without_key_score,
        "avg_reward": avg_reward
    }
    return results

step = ""
models = ["lstm", "mamba", "transformer"]
sizes = [4, 8 , 16]
models = ["mlp"]
sizes = [1]

total_results = {}

for model in models:
    for size in sizes:
        if model == "transformer" or model == "mamba":
            model_name = f"{model}_1layer"
        else:
            model_name = model
        embeddings_file = f"/home/alex/rl-starter-files/storage/MemoryNoops46_{step}{model_name}_seq{size}/MemoryNoops46_{step}{model_name}_seq{size}_export_map.json"
        result = linear_probe_wrapper(embeddings_file)
        total_results[f"{model}_{size}"] = result

pd.DataFrame(total_results).to_csv(f"linear_probe_results_mlp{step}.csv")



# embeddings_file = "/home/alex/rl-starter-files/storage/MemoryNoops46_4s_lstm_seq16/MemoryNoops46_4s_lstm_seq16_export_map.json"
# result = linear_probe_wrapper(embeddings_file)
# print(result)




