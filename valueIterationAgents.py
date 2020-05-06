# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import copy as cp

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        "*** YOUR CODE HERE ***"
        #calling the Value Iteration function passing iterations as argument
        self.valueIteration(self.iterations)

#function implementing value iteration
    def valueIteration(self,iterations):
        # Write value iteration code here
        #iterating from 0 to #iterations
        for i in range(0,iterations):
            #util.Couinter() dict to store the intermediate values
            intermediate_value = util.Counter()
            # Looping over all the states
            for state in self.mdp.getStates():
                #Check if it is a terminal state
                if self.mdp.isTerminal(state)==True:
                    #True then continue the loop
                    continue
                #else calculate
                else:
                    #List to push the values
                    vals = []
                    #considered highest value
                    best_value = -1*float('inf')
                    #get all the possible actions
                    actions = self.mdp.getPossibleActions(state)
                    #iterating over all the possible actions to generate the Q values
                    for action in actions:
                        q_value = self.computeQValueFromValues(state, action)
                        vals.append(q_value)
                        #Only consider the maximum Q value
                        best_value = max(vals)  
                        # Pass the best value to the dict
                        intermediate_value[state] = best_value
            # to copy all the contents of the local dict to global dict
            self.values = cp.copy(intermediate_value)
        
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        updated_Q_value = 0 #initiating to 0
        #getting the transition states and probabilities
        TSP = self.mdp.getTransitionStatesAndProbs(state, action)
        #temporary list to store the local value
        Q_sum = []
        #Iterating over all the states and action
        for next_state, probability in TSP:
            reward = self.mdp.getReward(state, action, next_state)
            discount = self.discount
            new = self.values[next_state]
            # Q(state, action) = SUM[ new_states) (probability(state, new_state) * (Reward + discount * V(new_state) ]
            updated_Q_value = probability * (reward + discount * new)
            #pushing the Q value to the list
            Q_sum.append(updated_Q_value)
        #getting the summation of the updated values
        updt = sum(Q_sum)
        #returning the updated value
        return updt
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #checking if it is a terminal state
        if self.mdp.isTerminal(state):
            #True then return None
            return None
        #else compute the decision
        else:
            #considering a maximum value
            best_option = -1*float('inf')
            #initiating the decision to None
            decision = None
            #local list
            vals = []
            #to store the actions from getPossibleActions of the passed state
            actions = self.mdp.getPossibleActions(state)
            #iterating over all the actions
            for action in actions:
                #calling the Q value function
                best_q_value = self.computeQValueFromValues(state, action)
                #Pushing it to the local list
                vals.append(best_q_value)
                #checking if it is the best value
                if best_option < best_q_value:
                    #If it is the best value opting the best value and its decision as the respective action
                    best_option, decision  = best_q_value, action
                #returning the best action as the decision
            return decision 
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
