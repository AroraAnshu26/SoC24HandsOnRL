import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class QAgent:
    def __init__(self, env: str) -> None:
        self.env_name = env
        self.env = gym.make(env) 
        self.state = self.env.reset()[0] # Variable to store current state of the environment
        
        self.observation_space_size = len(self.state) # 2 for Mountain Car, the x coordinate and the velocity of the cart
        
        self.actions = self.env.action_space.n # 3 for Mountain Car, represents total number of possible actions
        
        self.observation_space_low = self.env.observation_space.low # Returns array of length 2, representing the minimum values of position and velocity respectively. Consult documentation for more info.
        
        self.observation_space_high = self.env.observation_space.high
        
        # Hyperparameters. Play around with these values!!
        
        self.discrete_sizes = [50, 50] # Represents how many parts you want to discretize your observation space in. First element represents parts for position of car, and second for velocity of car.
        self.alpha = 0.3 # As defined in update rule
        self.gamma = 0.95 # As defined in update rule
        
        self.num_train_episodes = 2500 # Number of episodes to train the model for
        self.epsilon = 1 # Initial value for epsilon-greedy behavior
        self.num_episodes_decay = 150 # Number of episodes to act epsilon-greedily for, after which epsilon becomes 0
        self.epsilon_decay = self.epsilon / self.num_episodes_decay # Linear decay of epsilon, that is the amount to be decreased from epsilon after every episode termination
        
        '''
        Q-Table. We have provided one way to initialise it, and tried to keep it general, so you even try a different environment. You are adviced to think of other ways you could have initialised it.
        
        The dimensions of Q-Table must be parts of state-1 x parts of state-2 x actions (why?). It is initialised with random values here.
        
        * operator opens the array. So *[1,2] represents 1,2. Hence *self.discrete_sizes, self.actions represents 25, 25, 3 here
        the actions are accelerate to lef, no acc, acc to right
        '''
        self.q_table = np.random.uniform(low=-2, high=0, size=(*self.discrete_sizes, self.actions)) # Here I will get 25, 25, 3 which is 25 possible x, 25 possible velocity
                                                                                                    # Corresponding to each of them 3 possible actions, Q value-- expected reward in state (25*25)
                                                                                                    # And store all Q values corresponding to three actions.
    def get_state_index(self, state):                                                                
        '''
        Define a function which gives the index of the state in the Q_Table. Here a simple example to illustrate this task:
        
        Suppose low for position is 0, and high 2, and discretised it in 20 parts. Then the sections are [0-0.1], [0.1-0.2]...[1.9-2], and index for say position=0.45 will be 4 (in [0.4-0.5])
        
        The state here is an array of length self.observation_space_size(2). Other necessary variables are initialised in init method. Try to keep this function general for any environment, 
        but you may hardcode the numbers if you feel the task is difficult to generalise.
        
        Return a tuple containing the indices along each dimension
        '''
        state_indices = []
        
        for i, value in enumerate(state):
            # Calculate the range of the observation space for the current dimension
            range_val = self.observation_space_high[i] - self.observation_space_low[i]
            # Scale the value to be between 0 and 1
            scaled_value = (value - self.observation_space_low[i]) / range_val
            # Convert the scaled value to an index in the discretized space
            index = int(scaled_value * self.discrete_sizes[i])
            # Ensure the index is within bounds
            if index >= self.discrete_sizes[i]:
                index = self.discrete_sizes[i] - 1
            state_indices.append(index)
        
        return tuple(state_indices)
        

    def update(self, state, action, reward, next_state, is_terminal):
        '''
        Update the value of q[state, action] in the q-table based on the update rule. 
        First discretize both the state and next_state to get indices in q-table.
        The boolean is_terminal here represents whether the state action pair resulted in termination (NOT TRUNCATION) of environment.
        In this case, update the value by considering max_a' q(s', a,) = 0 (consult theory for why) and not based on q-table.
        '''
        state_idx = self.get_state_index(state)
        next_state_idx = self.get_state_index(next_state)

        if not is_terminal:
            future_q_value = np.max(self.q_table[next_state_idx])
        else:
            future_q_value = 0

        temporal_difference = reward + self.gamma * future_q_value - self.q_table[state_idx][action]

        self.q_table[state_idx][action] += self.alpha * temporal_difference


    def get_action(self):    
        '''
        Get the action either greedily, or randomly based on epsilon (You may use self.env.action_space.sample() to get a random action). Return an int representing action, based on self.state. Remember to discretize self.state first
        '''
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[self.get_state_index(self.state)]))
    
    
    def env_step(self):
        '''
        Takes a step in the environment and updated q-table
        '''
        action = self.get_action()
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        self.update(self.state, action, reward, next_state, terminated and not truncated) # terminated and not truncated is true when the episode got terminated but not truncated.
        
        self.state = next_state
        
        return terminated or truncated # Represents whether we need to reset the environment
    
    def agent_eval(self):
        '''Visualise the performance of agent'''
        eval_env = gym.make(self.env_name, render_mode="human")
        done = False
        eval_state = eval_env.reset()[0]
        while not done:
            action = int(np.argmax(self.q_table[self.get_state_index(eval_state)]))  # Take action based on greedy strategy now
            eval_state, reward, terminated, truncated, info = eval_env.step(action)
            
            eval_env.render() # Renders the environment on a window.
            
            done = terminated or truncated
            eval_state = eval_state  # Update eval_state with next state
        
    def train(self, eval_intervals):
        '''Main function to train the agent'''
        for episode in range(1, self.num_train_episodes + 1):
            done = False
            while not done:
                done = self.env_step()
            self.state = self.env.reset()[0] # Reset environment after end of episode
            
            self.epsilon = max(0, self.epsilon - self.epsilon_decay) # Update epsilon after every episode
            
            if episode % eval_intervals == 0:
                # Check performance of agent
                self.agent_eval()
        

if __name__ == "__main__":
    agent = QAgent("MountainCar-v0")
    agent.train(eval_intervals=10000) # Change the number to change frequency of evaluation
