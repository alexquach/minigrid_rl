import subprocess
import sys

def train_model(algo, env, model, recurrence, save_interval, frames, memory):
    # subprocess.run('bash -c "source activate root"', shell=True)
    command = [
        sys.executable, '-m', 'scripts.train',
        '--algo', algo,
        '--env', env,
        '--model', model,
        '--recurrence', str(recurrence),
        '--save-interval', str(save_interval),
        '--frames', str(frames),
        '--memory', memory
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    # Example configurations
    base_name = "MemoryNoops46"

    configurations = [
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_mamba_1layer_seq16", "recurrence": 16, "save_interval": 10, "frames": 10000000, "memory": "mamba"},
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_lstm_seq16", "recurrence": 16, "save_interval": 10, "frames": 10000000, "memory": "lstm"},
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_transformer_1layer_seq16", "recurrence": 16, "save_interval": 10, "frames": 10000000, "memory": "transformer"},
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_mamba_1layer_seq8", "recurrence": 8, "save_interval": 10, "frames": 10000000, "memory": "mamba"},
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_lstm_seq8", "recurrence": 8, "save_interval": 10, "frames": 10000000, "memory": "lstm"},
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_transformer_1layer_seq8", "recurrence": 8, "save_interval": 10, "frames": 10000000, "memory": "transformer"},
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_mamba_1layer_seq4", "recurrence": 4, "save_interval": 10, "frames": 10000000, "memory": "mamba"},
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_lstm_seq4", "recurrence": 4, "save_interval": 10, "frames": 10000000, "memory": "lstm"},
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_transformer_1layer_seq4", "recurrence": 4, "save_interval": 10, "frames": 10000000, "memory": "transformer"}
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_mamba_1layer_seq1", "recurrence": 1, "save_interval": 10, "frames": 10000000, "memory": "mamba"},
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_lstm_seq1", "recurrence": 1, "save_interval": 10, "frames": 10000000, "memory": "lstm"},
        # {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_transformer_1layer_seq1", "recurrence": 1, "save_interval": 10, "frames": 10000000, "memory": "transformer"},
        {"algo": "ppo", "env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_mlp_seq1", "recurrence": 1, "save_interval": 10, "frames": 10000000, "memory": "mlp"}
    ]

    for config in configurations:
        train_model(**config)
