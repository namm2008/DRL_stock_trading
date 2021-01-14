# DRL_stock_trading
Deep Reinforcement Learning for Stock trading task

## Abstract:
This project presented a new combination of Deep Reinforcement Learning algorithms. Prioritized Experience Replay which was proven to be robust in Atari Game revealed strong evidence to improve performance in the timeseries prediction task. In order to imitate the human-like learning instinct, Attention network was combined with encoding-decoding mechanism with the neural network to generate Q-values for the trading actions in this advanced DQN algorithm. The new model was compared with traditional EMA strategies and RNN models with Evaluation Metrics including Sharpe Ratio, Win Ratio, Maximum Drawdown/Return and Annualized Return. Inspiring result was obtained when daily data was used. 

## Deep Q-Learning 
### State and Action
The state design was one of the hardest tasks throughout the research. To concentrate on the research aims, the agent was set to only trade long and forbidden the short selling for simplicity reason. Also, the agent was only allowed to hold only one stock. In the replay memory, an additional column was added to show the holding stock status. A similar setting to Gao’s [4] of state and valid action pair was used. \
**Actions space: { ( 0 = Cash out or keep holding cash ), ( 1 = Buy a stock ) , ( 2 = keep holding a stock ) }**\
If the network output an invalid action, for example, when the agent was holding a stock, it was not supposed to buy another stock at the time. However, the network outputted 1 which violated the rule. In this case, the second highest value output would be chosen as the action. 
