# Thompson sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10

# Implementing Random Selection
import random
rand_ads_selecteds = []
rand_total_rewards = 0
for n in range(0,N):
    ad = random.randrange(d)
    rand_ads_selecteds.append(ad)
    reward = dataset.values[n, ad]
    rand_total_rewards = rand_total_rewards + reward
    
# Implementing Thompson sampling
import random
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
ts_ads_selecteds = []
ts_total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1,numbers_of_rewards_0[i] + 1)
        
        if random_beta > max_random:
            max_random = random_beta
            ad = i

    ts_ads_selecteds.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    ts_total_reward = ts_total_reward + reward

# Visualising the results
#ads_selecteds = rand_ads_selecteds
ads_selecteds = ts_ads_selecteds

plt.hist(ads_selecteds)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()