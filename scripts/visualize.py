import argparse
import numpy
import numpy as np

import utils
from utils import device
from tqdm import tqdm
import json

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default="video",
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=20,
                    help="number of episodes to visualize")
parser.add_argument("--memory", type=str, choices=['lstm', 'mamba', 'transformer'], default=None,
                    help="type of memory module to use: lstm | mamba | transformer (default: None)")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--filename", type=str, default="status.pt",
                    help="filename of the model to load")

args = parser.parse_args()

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
print(f"Device: {device}\n")

# Load environment
env = utils.make_env(args.env, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir, filename=args.filename,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent
if args.gif:
    from array2gif import write_gif
    args.gif = f"{model_dir}/{args.model}_{args.gif}"

    frames = []

# Create a window to view the environment
env.render()
export_map = []

for episode in tqdm(range(args.episodes), desc="Visualizing Episodes"):
    obs, _ = env.reset()

    timestep = 0
    has_opened_red_door = False
    while True:
        env.render()
        if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))

        action, embedding = agent.get_action(obs)
        direction = obs.get("direction")
        if [4, 0, 0] in obs['image'].reshape(-1, 3).tolist():
            has_opened_red_door = True
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done)

        export_map.append({
            "episode": episode,
            "timestep": timestep,
            "action": action,
            "embedding": embedding,
            "reward": reward,
            "done": done,
            "direction": direction,
            "redOpened": has_opened_red_door
        })
        if done:
            break
        timestep += 1

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if args.filename == "status.pt":
    export_map_filename = f"{model_dir}/{args.model}_export_map.json"
else:
    export_map_filename = f"{model_dir}/{args.model}_export_map_{args.filename}.json"
with open(export_map_filename, 'w') as f:
    json.dump(export_map, f, indent=4, cls=NpEncoder)
print(f"Export map saved to {export_map_filename}")


if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
