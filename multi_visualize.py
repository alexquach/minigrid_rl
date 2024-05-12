import subprocess
import sys

def visualize_model(env, model, memory, seqlen):
    command = [
        'python3', '-m', 'scripts.visualize',
        '--env', env,
        '--model', model,
        '--memory', memory,
        '--seqlen', str(seqlen)
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    # Example configurations
    base_name = "MemoryNoops46"

    configurations = [
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_mamba_1layer_seq16", "memory": "mamba", "seqlen": 16},
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_lstm_seq16", "memory": "lstm", "seqlen": 16},
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_transformer_1layer_seq16", "memory": "transformer", "seqlen": 16},
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_mamba_1layer_seq8", "memory": "mamba", "seqlen": 8},
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_lstm_seq8", "memory": "lstm", "seqlen": 8},
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_transformer_1layer_seq8", "memory": "transformer", "seqlen": 8},
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_mamba_1layer_seq4", "memory": "mamba", "seqlen": 4},
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_lstm_seq4", "memory": "lstm", "seqlen": 4},
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_transformer_1layer_seq4", "memory": "transformer", "seqlen": 4}
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_mamba_1layer_seq1", "memory": "mamba", "seqlen": 1},
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_lstm_seq1", "memory": "lstm", "seqlen": 1},
        # {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_transformer_1layer_seq1", "memory": "transformer", "seqlen": 1},
        {"env": "MiniGrid-MemoryS7-v0", "model": f"{base_name}_mlp_seq1", "memory": "mlp", "seqlen": 1},
    ]

    for config in configurations:
        visualize_model(**config)
