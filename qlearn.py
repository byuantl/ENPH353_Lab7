import random
import pickle
import os
import csv

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.

        try:
            full_filename = filename + ".pickle"
            
            # Check if file exists
            if not os.path.exists(full_filename):
                print(f"File {full_filename} not found!")
                return
            
            with open(full_filename, 'rb') as f:
                loaded_q = pickle.load(f)
            
            self.q.update(loaded_q)
            
            print("Loaded file: {}".format(filename+".pickle"))
            
        except Exception as e:
            print(f"Error loading Q values from {filename}.pickle: {e}")

        

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        try:
            # Save pickle
            pickle_filename = filename + ".pickle"
            with open(pickle_filename, 'wb') as f:
                pickle.dump(self.q, f)
                print("Wrote to file: {}".format(pickle_filename))

            
            # Save csv
            csv_filename = filename + ".csv"
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(['state', 'action', 'q_value'])
                
                # Q-value entry
                for (state, action), q_value in self.q.items():
                    state_str = ''.join(str(x) for x in state)
                    writer.writerow([state_str, action, q_value])

                print("Wrote to file: {}".format(csv_filename))
            
        except Exception as e:
            print(f"Error saving Q values: {e}")


    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        # Exploration: random action
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            q_value = self.getQ(state, action)
        # Exploitation: best action
        else:
            # Get Q values for all actions from this state
            q_values = [self.getQ(state, a) for a in self.actions]
            max_q = max(q_values)
            
            # Find all actions that have the maximum Q value
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            
            # Randomly choose among the best actions to break ties
            action = random.choice(best_actions)
            q_value = max_q
        
        if return_q:
            return action, q_value
        else:
            return action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        current_q = self.getQ(state1, action1)
        
        max_next_q = max([self.getQ(state2, a) for a in self.actions])
        
        # Bellman update
        self.q[(state1,action1)] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    
