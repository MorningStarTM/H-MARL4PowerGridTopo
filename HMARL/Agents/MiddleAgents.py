import numpy as np
import random


class FixedSubPicker(object):
    """_summary_
    This class defines the core functionality for selecting substations in a specific order.
    """
    
    def __init__(self, masked_sorted_sub, **kwargs):
        """_summary_
        Initializes the class when a new FixedSubPicker object is created.
        """
        self.masked_sorted_sub = masked_sorted_sub # list of substations to consider for activation
        self.subs_2_act = [] #  empty list that will be used to store the order in which substations will be acted upon.
        n_subs = len(masked_sorted_sub) # number of substations
        self.count = np.zeros((n_subs, n_subs), int) # NumPy zero-filled matrix of size n_subs x n_subs (number of substations by number of substations). 
        #It's potentially used for tracking the frequency of transitions between substation selections
        self.previous_sub = -1 #  index indicating the last substation that was selected, initially set to -1 (indicating no prior selection).
        self.prev_sub = None

    def complete_reset(self):
        """_summary_
        Resets the class for a new episode or simulation run
        """
        self.subs_2_act = [] # Clears the subs_2_act list, which holds the order of substations to be processed.
        self.previous_sub = -1 # resets previous_sub to -1 again, indicating no previous selection.

    def pick_sub(self, obs, sample):
        """_summary_
        Core component of the FixedSubPicker class 
        responsible for selecting substations in the middle layer of the MARL system for power grid optimization
        """
        # 1. Check if the List of Substations to Act Upon is Empty
        if len(self.subs_2_act) == 0:
            # If it is, the list is reinitialized with the masked_sorted_sub, which contains the substations in a predefined order.
            # randomize the order in which the substations are activated   
            self.subs_2_act = list(self.masked_sorted_sub)
        
        # 2. Select the Next Substation 
        sub_2_act = self.subs_2_act.pop(0) # selects the first substation from the subs_2_act list 
        
        # 3. Update Transition Count and Previous Substation (if sample is True)
        if sample:
            self.count_transitions(sub_2_act) # updates the transition count matrix to reflect the transition from the previous substation to the newly selected substation.
            self.previous_sub = sub_2_act # keeping track of the last substation that was selected.
            self.prev_sub = sub_2_act
        return sub_2_act # returns sub_2_act, the substation to be acted upon next.


# Track and calculate the transition probabilities between substation selections in the middle-level agent
    def count_transitions(self, next_sub):
        """_summary_
        updates the count matrix, which keeps track of how often substations are selected one after another
        """
        if self.previous_sub >= 0: # Checks whether a valid previous substation exists
            
            # Finds the indices of the previous substation (prev) and 
            # the current substation (next_sub) in the masked_sorted_sub list using np.flatnonzero.
            prev = np.flatnonzero(self.masked_sorted_sub == self.previous_sub).squeeze()
            next = np.flatnonzero(self.masked_sorted_sub == next_sub).squeeze()
            self.count[prev, next] += 1 # Increments the value in the count matrix at the position (prev, next)

    
 
    # Calculates the probability of transitioning between substations in the middle-level agent. 
    @property
    def transition_probs(self):
        row_sums = self.count.sum(axis=1, keepdims=True)
        non_zero_rows = (row_sums != 0).squeeze()
        probs = np.zeros_like(self.count, float)
        probs[non_zero_rows] = self.count[non_zero_rows] / row_sums[non_zero_rows]
        return probs



class RuleBasedSubPicker(FixedSubPicker):
    """_summary_
    Responsible for selecting the substations to act upon and determining
    their order based on the CAPA policy
    """
    def __init__(self, masked_sorted_sub, action_space):
        """_summary_
        Initializes the RuleBasedSubPicker instance.
        """
        # 1. Call the Base Class Constructor
        super().__init__(masked_sorted_sub) # calls the __init__ method of the parent class (FixedSubPicker)
        
        # 2. Initialize Substation Line Information
        self.sub_line_or = [] # This list will store the indices of outgoing lines (lines leaving the substation) for each substation in masked_sorted_sub.
        self.sub_line_ex = [] # This list will store the indices of incoming lines (lines entering the substation) for each substation.
        
        # 3. Looping through substations and appends the indices of lines originating from (line_or_to_subid) and ending at (line_ex_to_subid) that substation to the corresponding lists
        for sub in self.masked_sorted_sub:
            self.sub_line_or.append(
                np.flatnonzero(action_space.line_or_to_subid == sub)
            )
            self.sub_line_ex.append(
                np.flatnonzero(action_space.line_ex_to_subid == sub)
            )

    def pick_sub(self, obs, sample):
        """_summary_
        Selects substations to act upon based on specific rules.
        It works with data related to power grid substations and aims to 
        identify which substation requires urgent care based on line utilization.
        
        """
        
        # 1. Checking for Existing Substation List:
        if len(self.subs_2_act) == 0: # checks if the self.subs_2_act list is empty.
            # rule c: CAPA gives high priority to substations with lines under high utilization of their capacity,
            # which applies an action to substations that require urgent care.
            # If the list is empty (meaning there are no pre-defined substations to choose from), 
            # it indicates the beginning of a new episode, and the selection process needs to be initiated.
            
            rhos = [] # stores the calculated rho values for each substation.
            
            # 2. Looping through Substations and Analysing Line Utilization
            for sub in self.masked_sorted_sub:
                sub_i = np.flatnonzero(self.masked_sorted_sub == sub).squeeze() # finds the index (sub_i) of the current substation (sub) within the masked_sorted_sub list.
                
                rho = np.append(
                    obs.rho[self.sub_line_or[sub_i]].copy(),
                    obs.rho[self.sub_line_ex[sub_i]].copy(),
                )
                
                
                
                rho[rho == 0] = 3
                rho_max = rho.max()
                rho_mean = rho.mean()
                rhos.append((rho_max, rho_mean))
            order = sorted(
                zip(self.masked_sorted_sub, rhos), key=lambda x: (-x[1][0], -x[1][1])
            )
            self.subs_2_act = list(list(zip(*order))[0])
        sub_2_act = self.subs_2_act.pop(0)
        if sample:
            self.count_transitions(sub_2_act)
            self.previous_sub = sub_2_act
        self.prev_sub = sub_2_act
        return sub_2_act
