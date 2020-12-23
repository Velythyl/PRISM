from data_collection import data_collection
from env import NormalEnv, ChromShiftEnv, EncapsulatePrismEnv
from test import evaluate
from train import train
from train_prism import train_prism

env_name = "PongNoFramekip-v4"

# 1. Train Alice and Bob
alice = train(env_name, 1000000, 6, NormalEnv, "CnnPolicy")
bob = train(env_name, 1000000, 6, ChromShiftEnv, "CnnPolicy")

# 2. Evaluate Alice and Bob
print("Alice's performance:", evaluate(NormalEnv(env_name, 1), 200, alice, False))
print("Bob's performance:", evaluate(ChromShiftEnv(env_name, 1), 200, bob, False))

# 3. Collect Data
data_collection(env_name, alice, bob, NormalEnv(env_name, 1), ChromShiftEnv(env_name, 1), 20000)

# 4. Train Prism
prism = train_prism(env_name)
prism.eval()

# 5. Train Charlie
prism_env_fun = EncapsulatePrismEnv(prism, is_shifted=False)
charlie = train(env_name, 21500000, 6, prism_env_fun, "MlpPolicy", policy_kwargs={"net_arch": []})

# 6. Evaluate Charlie
print("Charlie's performance on her own env:", evaluate(prism_env_fun(env_name, 1), 200, charlie, False))
print("Charlie's performance on the shifted prismed env:", evaluate(EncapsulatePrismEnv(prism, is_shifted=True)(env_name, 1), 200, charlie, False))
