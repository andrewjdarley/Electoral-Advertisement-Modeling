# Presidential Election Ad Spending Model

This repository contains a statistical model analyzing how campaign advertising spending affects election outcomes per state in United States presidential elections between 2012 and 2024

## Overview

The model fits a <a href='https://en.wikipedia.org/wiki/Beta_distribution'>Beta Distribution</a> whose parameters, a and b, are determine by the following:

- Historical partisan lean of each state
- Campaign advertising spending by both parties
- State population

## Model Structure

For each state, the Democratic vote share (proportion of votes for the Republican or Democratic candidate that are Democratic) is modeled as a Beta distribution with parameters:

$$a_s = \frac{g(D_s)}{N_s} + p_{D,s}$$

$$b_s = \frac{h(R_s)}{N_s} + p_{R,s}$$

Where

- $D_s$: amount of money invested (in millions) in the state by Democrats

- $R_s$: amount of money invested (in millions) in the state by Republicans

- $N_s$: state's inverse distance weighted average (distance = number of elections away) of number of Partisan (Republican or Democrat) voters

- $p_{D,s}$: state's historical Democratic advantage

- $p_{R,s}$: state's historical Republican advantage

I defined advantage by

- $p_{D,s} = k \cdot \pi_s$
- $p_{R,s} = k \cdot (1 - \pi_s)$

where $\pi_s$ is the weighted democratic share over the past 3 years and $k$ is a variable parameter that determines how much states stick to their history with high values representing lower elasticity

The spending functions $g$ and $h$ consistent across all states are linear, satisfy $g(0) = h(0) = 0$, and are monotonically increasing

The model parameters that yielded the greatest likelihood were: 

$k = 300$
 
$g(D) = 4.318453 \times 10^5 \cdot D$

$h(R) = 1.443381 \times 10^6 \cdot R$

## Files

- **`final_model.ipynb`** - Final Beta distribution model with spendingeffect
- **`difference_model.ipynb`** - Earlier approach modeling turnout changes from prior years
- **`eda.ipynb`** - Exploratory data analysis and visualizations
- **`ad_data.csv`** - Campaign spending data and election results by state and year
- **`countypres_2000-2024.csv`** - County-level presidential election results

## Key Findings

- Republican Ad spending was more efficient than Democrat ad spending in this
 model
- States with little ad spending show higher outcome variance
- Historical partisan strongly predicts election outcomes
- The model achieves good fit with simple linear functions after accounting for historical partisan advantage

## Requirements
```
pandas
numpy
matplotlib
scipy
seaborn
```

## References
https://www.democracyinaction.us/2020/states/advertisingoverview.html
https://www.nbcnews.com/politics/first-read/pro-clinton-battleground-ad-spending-outstrips-trump-team-2-1-n677911
https://www.capradio.org/articles/2024/11/01/more-than-10-billion-has-been-spent-on-ads-in-the-2024-election/
https://abcnews.go.com/Politics/OTUS/election-2012-campaigns-numbers/story?id=17647443&utm_source=chatgpt.com
https://en.wikipedia.org/wiki/Beta_distribution
 MIT Election Data and Science Lab. 2018. “County Presidential Election Returns 2000-2024.” Harvard Dataverse. https://doi.org/10.7910/DVN/VOQCHQ.

(2012 data was estimated by assuming that Obama dominated 2/3 of the ad market)