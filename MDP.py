import numpy as np
import random
import collections
import math

DEBUG = False
def debug(str):
    if DEBUG:
        print(str)       


class Agent:
    def __init__(self, No_states, initial_state):
        """
        Initialize the agent with the number of states, the initial state, 
        the maximum number of iterations, discount factor and the threshold.
        
        Parameters:
            No_states (int): The number of states in the MDP
            initial_state (int): The initial state of the agent
        """
        self.initial_state_ = initial_state
        self.numstates_ = No_states
        self.max_iterations = 100
        self.discount_factor = 0.95
        self.threshold = 1e-4

    def states(self, No_states):
        """
        Return a list of states given the number of states
        
        Parameters:
            No_states (int): The number of states in the MDP
        
        Returns:
            list: A list of integers representing the states in the MDP
        """
        return list(np.arange(1, No_states))

    def initiate_values(self, No_states):
        """
        Return an array of zeros with shape (No_states, 1)
        
        Parameters:
            No_states (int): The number of states in the MDP
        
        Returns:
            numpy.ndarray: A 2D array of zeros with shape (No_states, 1)
        """
        return np.zeros((No_states, 1))
    
    def initiate_reward(self, No_states):
        """
        Return an array of random values with shape (No_states, No_states)
        
        Parameters:
            No_states (int): The number of states in the MDP
        
        Returns:
            numpy.ndarray: A 2D array of random values with shape (No_states, No_states)
        """
        return np.random.rand(No_states, No_states)

    def initiate_policy(self, No_states):
        """
        Return an array of random integers with shape (No_states, 1)
        
        Parameters:
            No_states (int): The number of states in the MDP
        
        Returns:
            numpy.ndarray: A 2D array of random integers with shape (No_states, 1)
        """
        return np.random.randint(0, No_states, (No_states, 1))

    def transition_(self, No_states):

        """
        Perform value iteration and policy iteration to find the optimal value function and policy.
        
        Parameters:
            No_states (int): The number of states in the MDP
        
        Returns:
            tuple: A tuple containing the optimal value function and policy, both represented as numpy.ndarray
        
        """
        states = list(np.arange(1, self.numstates_+1))
        debug("states present are")
        debug(states)

        transition_probs = np.random.rand(No_states, No_states)
        transition_probs = transition_probs / transition_probs.sum(axis=1, keepdims=True)

        reward_func = self.initiate_reward(No_states)
        value_func = self.initiate_values(No_states)
        policy_func = self.initiate_policy(No_states)

        # Value iteration
        for i in range(self.max_iterations):
            previous_value_matrix = value_func.copy()
            for state in range(self.numstates_):
                value_func[state] = np.max(reward_func[state] + self.discount_factor * transition_probs[state] @ previous_value_matrix)
            if np.max(np.abs(previous_value_matrix - value_func)) < self.threshold:
                break
        
        #policy iteration
        for i in range(self.max_iterations):
            previous_policy_matrix = policy_func.copy()
            for state in range(self.numstates_):
                action_values = reward_func[state] + self.discount_factor * transition_probs[state] @ value_func
                policy_func[state] = np.argmax(action_values)
            if (previous_policy_matrix == policy_func).all():
                break

        debug("Randomly initiated value function:{}\n Reward for all the states: {}".format(reward_func, value_func))
        no_transitions = int(input("Please enter number of transitions the Agent should take"))
        return value_func, policy_func

No_states = int(input("Enter the number of states"))
initial_state = int(input("Enter the initial state"))

RL = Agent(No_states = No_states, initial_state = initial_state)
value_matrix, policy_matrix = RL.transition_(No_states = No_states)
print("Value Matrix after iteration:\n {}\n Policy Matrix after iterations:\n {} ".format(value_matrix, policy_matrix))
