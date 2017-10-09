# Upper Confidence Bound

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

# Implementing Upper Confidence Bound
ucb_ads_selected = integer(0)
numbers_of_selections = integer(d)
sums_of_rewards = integer(d)
ucb_total_reward = 0
for (n in 1:N) {
  max_ucb = 0
  ad = 0

  for (i in 1:d) {
    if (numbers_of_selections[i] > 0) {
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]
      delta_i = sqrt(3/2 * log(n) / numbers_of_selections[i])
      ucb = average_reward + delta_i  
    } else {
      # set ucb to very large number -> make sure all value from 1 to d was selected
      ucb = 1e400
    }

    if (ucb > max_ucb) {
      max_ucb = ucb
      ad = i
    }
  }
  
  ucb_ads_selected = append(ucb_ads_selected, ad)
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  ucb_total_reward = ucb_total_reward + reward
}

# Visualising the results
# ads_selected = random_ads_selected
ads_selected = ucb_ads_selected
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')