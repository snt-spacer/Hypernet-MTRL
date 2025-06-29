import torch
from rsl_rl.modules.actor_critic_memory import ActorCriticMemory
import matplotlib.pyplot as plt

path = "/workspace/isaaclab/logs/rsl_rl/multitask_memory/2025-06-20_07-34-58_rsl-rl_ppo_memory_GoToPosition-GoToPose-TrackVelocities_Turtlebot2_r-0_seed-42/model_1000.pt"

# Define the observation and action spaces (replace with your actual dimensions)
num_obs = 13
num_privileged_obs = 4
num_actions = 2
num_tasks = 2
device = "cuda:0"

# Instantiate the ActorCriticMemory model
actor_critic = ActorCriticMemory(
    num_actor_obs=num_obs,
    num_critic_obs=num_obs,
    num_actions=num_actions,
    actor_hidden_dims=[32,32],
    critic_hidden_dims=[32,32],
    activation="elu",
    init_noise_std=1.0,
    clip_actions=True,
    clip_actions_range=(0.0, 1.0),
    use_embeddings=True,
    embeddings_size=32,
    generator_size=(64,64),
    num_memory_obs=num_privileged_obs,
    network_type="hybrid", #pure, hybrid
)

# Load the model weights
loaded_dict = torch.load(path)
actor_critic.load_state_dict(loaded_dict["model_state_dict"])
actor_critic.to(device)

# Print the model to confirm
print(actor_critic)
print("-"*20)

# Print parameter names and shapes
for name, param in actor_critic.named_parameters():
    print(f"{name}: {param.shape}")

# Print parameter names, shapes, and some sample values
for name, param in actor_critic.named_parameters():
    print(f"{name}: {param.shape}")
    print(f"  Sample values: {param.flatten()[:5]}")  # First 5 values
    print(f"  Mean: {param.mean():.6f}, Std: {param.std():.6f}")
    print()

print("-"*20)

# Check actor network weights - adapted for PureMemoryActorNetwork
print("Actor network structure:")
print(f"Actor type: {type(actor_critic.actor)}")
print(f"Actor: {actor_critic.actor}")

# Check if actor has specific components
if hasattr(actor_critic.actor, 'named_parameters'):
    print("\nActor network parameters:")
    for name, param in actor_critic.actor.named_parameters():
        print(f"  {name}: {param.shape}")
        print(f"    Sample values: {param.flatten()[:5]}")
        print(f"    Mean: {param.mean():.6f}, Std: {param.std():.6f}")

# Check critic network weights  
print("\nCritic network weights:")
if hasattr(actor_critic.critic, 'named_parameters'):
    for name, param in actor_critic.critic.named_parameters():
        print(f"  {name}: {param.shape}")
        print(f"    Sample values: {param.flatten()[:5]}")

# Check embedding weights
if hasattr(actor_critic, 'embedding'):
    print("\nEmbedding weights:")
    if hasattr(actor_critic.embedding, 'weight'):
        print(f"Embedding shape: {actor_critic.embedding.weight.shape}")
        print(f"Values:\n{actor_critic.embedding.weight}")

# Check generator weights
if hasattr(actor_critic, 'generator'):
    print("\nGenerator weights:")
    if hasattr(actor_critic.generator, 'named_parameters'):
        for name, param in actor_critic.generator.named_parameters():
            print(f"  {name}: {param.shape}")
            print(f"    Values:\n{param}")

# Check memory components
if hasattr(actor_critic, 'memory_a'):
    print("\nMemory actor components:")
    print(f"Memory actor: {actor_critic.memory_a}")
    if hasattr(actor_critic.memory_a, 'named_parameters'):
        for name, param in actor_critic.memory_a.named_parameters():
            print(f"  {name}: {param.shape}")

if hasattr(actor_critic, 'memory_c'):
    print("\nMemory critic components:")
    print(f"Memory critic: {actor_critic.memory_c}")
    if hasattr(actor_critic.memory_c, 'named_parameters'):
        for name, param in actor_critic.memory_c.named_parameters():
            print(f"  {name}: {param.shape}")

# Check if weights are non-zero (indicating they were loaded)
print("\nNon-zero weight analysis:")
for name, param in actor_critic.named_parameters():
    if param.requires_grad:
        non_zero = (param != 0).sum().item()
        total = param.numel()
        print(f"{name}: {non_zero}/{total} non-zero values ({100*non_zero/total:.1f}%)")

def analyze_weights(model):
    """Analyze weight statistics for all layers"""
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only trained parameters
            print(f"\n{name}:")
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {param.mean():.6f}")
            print(f"  Std: {param.std():.6f}")
            print(f"  Min: {param.min():.6f}")
            print(f"  Max: {param.max():.6f}")
            print(f"  Norm: {param.norm():.6f}")

print("\n" + "="*50)
print("DETAILED WEIGHT ANALYSIS")
print("="*50)
analyze_weights(actor_critic)

# Create a fresh model to compare
print("\n" + "="*50)
print("COMPARISON WITH FRESH MODEL")
print("="*50)
fresh_model = ActorCriticMemory(
    num_actor_obs=num_obs,
    num_critic_obs=num_obs,
    num_actions=num_actions,
    actor_hidden_dims=[32,32],
    critic_hidden_dims=[32,32],
    activation="elu",
    init_noise_std=1.0,
    clip_actions=True,
    clip_actions_range=(-1.0, 1.0),
    use_embeddings=True,
    embeddings_size=32,
    generator_size=(64, 64),
    num_memory_obs=num_privileged_obs,
    network_type="hybrid", #pure, hybrid
).to(device)

# Compare weights
for name, loaded_param in actor_critic.named_parameters():
    if name in fresh_model.state_dict():
        fresh_param = fresh_model.state_dict()[name]
        diff = (loaded_param - fresh_param).abs().mean()
        print(f"{name}: Mean difference from random = {diff:.6f}")

def plot_module_weight_distributions(module, module_name="module"):
    """Plot weight distributions for all layers of a given module (actor/critic)."""
    import matplotlib.pyplot as plt
    params = [p for p in module.named_parameters() if p[1].requires_grad]
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    for i, (name, param) in enumerate(params):
        axes[i].hist(param.flatten().detach().cpu().numpy(), bins=50, alpha=0.7)
        axes[i].set_title(f'{module_name}: {name}\nMean: {param.mean():.4f}, Std: {param.std():.4f}')
        axes[i].set_xlabel('Weight Value')
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"source/isaaclab_tasks/isaaclab_tasks/rans/utils/{module_name}_weight_distributions.png")
    plt.close(fig)

print("\n" + "="*50)
print("PLOTTING WEIGHT DISTRIBUTIONS")
print("="*50)
plot_module_weight_distributions(actor_critic.actor, module_name="actor")
plot_module_weight_distributions(actor_critic.critic, module_name="critic")


# Check if the model has the expected components
print("\n" + "="*50)
print("MODEL COMPONENTS")
print("="*50)
print("Model components:")
print(f"Actor network: {actor_critic.actor}")
print(f"Critic network: {actor_critic.critic}")
print(f"Embedding: {actor_critic.embedding if hasattr(actor_critic, 'embedding') else 'None'}")
print(f"Generator: {actor_critic.generator if hasattr(actor_critic, 'generator') else 'None'}")
print(f"Memory actor: {actor_critic.memory_a if hasattr(actor_critic, 'memory_a') else 'None'}")
print(f"Memory critic: {actor_critic.memory_c if hasattr(actor_critic, 'memory_c') else 'None'}")

# Check action distribution parameters
print("\nAction distribution parameters:")
if hasattr(actor_critic, 'std'):
    print(f"Action std: {actor_critic.std}")
if hasattr(actor_critic, 'mean'):
    print(f"Action mean: {actor_critic.mean}")
if hasattr(actor_critic, 'action_std'):
    print(f"Action std (alternative): {actor_critic.action_std}")
if hasattr(actor_critic, 'action_mean'):
    print(f"Action mean (alternative): {actor_critic.action_mean}")

