# Upper Confidence Bound

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
    
# Implementing UCB
import math
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
ucb_ads_selecteds = []
ucb_total_reward = 0
for n in range(0, N):
    ad = 0
    max_ucb = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            ucb = average_reward + delta_i
        else:
            ucb = 10e400 # set ucb to very large number -> make sure all value from 1 to d was selected
        if ucb > max_ucb:
            max_ucb = ucb
            ad = i
    ucb_ads_selecteds.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    ucb_total_reward = ucb_total_reward + reward

# Visualising the results
#ads_selecteds = rand_ads_selecteds
ads_selecteds = ucb_ads_selecteds

plt.hist(ads_selecteds)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()