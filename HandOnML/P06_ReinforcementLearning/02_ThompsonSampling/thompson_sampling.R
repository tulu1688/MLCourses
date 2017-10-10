# Thompson sampling

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10

# Implementing Random Selection
random_ads_selected = integer(0)
random_total_reward = 0
for (n in 1:N) {
  ad = sample(1:d, 1)
  random_ads_selected = append(random_ads_selected, ad)
  reward = dataset[n,ad]
  random_total_reward = random_total_reward + reward
}

# Implementing Thompson sampling
ts_ads_selected = integer(0)
numbers_of_rewards_1 = integer(d)
numbers_of_rewards_0 = integer(d)
ts_total_reward = 0
for (n in 1:N) {
  max_random = 0
  ad = 0
  
  for (i in 1:d) {
    random_beta = rbeta(n = 1,
                        shape1 = numbers_of_rewards_1[i] + 1,
                        shape2 = numbers_of_rewards_0[i] + 1)
    
    if (random_beta > max_random) {
      max_random = random_beta
      ad = i
    }
  }
  
  ts_ads_selected = append(ts_ads_selected, ad)
  reward = dataset[n, ad]
  if (reward == 1) {
    numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
  } else {
    numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
  }
  ts_total_reward = ts_total_reward + reward
}

# Visualising the results
# ads_selected = random_ads_selected
ads_selected = ts_ads_selected
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')