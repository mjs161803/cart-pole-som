# This class is able to receive observations (as NumPy arrays), and returns a binary signal for 'goal_state' 
# indicating whether the received observation is an element of the subset of the input space mapping to 'goal_state'
# The class has a private method named 'assess_state(observation)' which holds the algorithm for defining the goal state space.
# The class also maintains private members to collect statistics about how frequently the goal state is observed,
# by keeping a running tally of how many observations it has assessed, and how often the goal_state is observed.

import numpy as np


class AgentTeacher:
    """
    A teacher class that assesses observations and determines if they are in the goal state.
    """
    
    def __init__(self):
        """Initialize the teacher with statistics counters."""
        self._total_observations = 0
        self._goal_state_count = 0
    
    def assess_state(self, observation):
        """
        Private method that determines if an observation is in the goal state.
        
        Args:
            observation: A NumPy array representing the current state.
        
        Returns:
            bool: True if the observation is in the goal state, False otherwise.
        """
        # Define goal state criteria 
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
        
        # Goal state: pole nearly vertical and cart near center
        goal_achieved = (
            abs(pole_angle) < 0.1 # and
            # abs(pole_angular_velocity) < 0.5 and
            # abs(cart_position) < 2.4 and
            #abs(cart_velocity) < 0.5
        )
        
        return goal_achieved
    
    def evaluate(self, observation):
        """
        Public method to evaluate if an observation is in the goal state.
        Updates statistics about observed goal states.
        
        Args:
            observation: A NumPy array representing the current state.
        
        Returns:
            bool: True if the observation is in the goal state, False otherwise.
        """
        self._total_observations += 1
        goal_state = self.assess_state(observation)
        
        if goal_state:
            self._goal_state_count += 1
        
        return goal_state
    
    def _assess_state(self, observation):
        """Private wrapper for assess_state to match naming convention."""
        return self.assess_state(observation)
    
    def get_goal_state_frequency(self):
        """
        Returns the frequency of goal state observations.
        
        Returns:
            float: Ratio of goal state observations to total observations.
        """
        if self._total_observations == 0:
            return 0.0
        return self._goal_state_count / self._total_observations
    
    def get_statistics(self):
        """
        Returns statistics about observations.
        
        Returns:
            dict: Dictionary containing total observations and goal state count.
        """
        return {
            'total_observations': self._total_observations,
            'goal_state_count': self._goal_state_count,
            'goal_state_frequency': self.get_goal_state_frequency()
        }

