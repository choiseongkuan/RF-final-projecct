import matplotlib.pyplot as plt
import pickle

with open("data3.pkl", "rb") as f:
    x, y = pickle.load(f)

plt.plot(x, y)
plt.title("HalfCheetah")
plt.xlabel("timestep")
plt.ylabel("reward")
plt.savefig("result/Improved_PPO.png")
plt.show()