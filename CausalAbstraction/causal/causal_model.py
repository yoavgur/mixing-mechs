import random, sys
# from pyvis.network import Network
# import webbrowser, os, networkx as nx
# import json

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import copy
import itertools
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from datasets import Dataset, load_dataset

from causal.counterfactual_dataset import CounterfactualDataset


class CausalModel:
    """
    A class to represent a causal model with variables, values, parents, and mechanisms.
    Attributes:
    -----------
    variables : list
        A list of variables in the causal model.
    values : dict
        A dictionary mapping each variable to its possible values.
    parents : dict
        A dictionary mapping each variable to its parent variables.
    mechanisms : dict
        A dictionary mapping each variable to its causal mechanism.
    print_pos : dict, optional
        A dictionary specifying positions for plotting (default is None).
    """
    def __init__(
        self,
        variables,
        values,
        parents,
        mechanisms,
        print_pos=None,
        id="null"
    ):
        """
        Initialize a CausalModel instance with the given parameters.
        
        Parameters:
        -----------
        variables : list
            A list of variables in the causal model.
        values : dict
            A dictionary mapping each variable to its possible values.
        parents : dict
            A dictionary mapping each variable to its parent variables.
        mechanisms : dict
            A dictionary mapping each variable to its causal mechanism.
        print_pos : dict, optional
            A dictionary specifying positions for plotting (default is None).
        """
        self.variables = variables
        self.values = values
        self.parents = parents
        self.mechanisms = mechanisms
        self.id = id
        assert "raw_input" in self.variables, "Variable 'raw_input' must be present in the model variables."
        assert "raw_output" in self.variables, "Variable 'raw_output' must be present in the model variables."

        # Create children and verify model integrity
        self.children = {var: [] for var in variables}
        for variable in variables:
            assert variable in self.parents
            for parent in self.parents[variable]:
                self.children[parent].append(variable)

        # Find inputs and outputs
        self.inputs = [var for var in self.variables if len(parents[var]) == 0]
        self.outputs = copy.deepcopy(variables)
        for child in variables:
            for parent in parents[child]:
                if parent in self.outputs:
                    self.outputs.remove(parent)

        # Generate timesteps
        self.timesteps = {input_var: 0 for input_var in self.inputs}
        step = 1
        change = True
        while change:
            change = False
            copytimesteps = copy.deepcopy(self.timesteps)
            for parent in self.timesteps:
                if self.timesteps[parent] == step - 1:
                    for child in self.children[parent]:
                        copytimesteps[child] = step
                        change = True
            self.timesteps = copytimesteps
            step += 1
        self.end_time = step - 2
        for output in self.outputs:
            self.timesteps[output] = self.end_time

        # Verify that the model is valid
        for variable in self.variables:
            try:
                assert variable in self.values
            except AssertionError:
                raise ValueError(f"Variable {variable} not in values")
            try:
                assert variable in self.children
            except AssertionError:
                raise ValueError(f"Variable {variable} not in children")
            try:
                assert variable in self.mechanisms
            except AssertionError:
                raise ValueError(f"Variable {variable} not in mechanisms")
            try:
                assert variable in self.timesteps
            except AssertionError:
                raise ValueError(f"Variable {variable} not in timesteps")

            for variable2 in copy.copy(self.variables):
                if variable2 in self.parents[variable]:
                    try:
                        assert variable in self.children[variable2]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable} not in children of {variable2}")
                    try:
                        assert self.timesteps[variable2] < self.timesteps[variable]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable2} has a later timestep than {variable}")
                if variable2 in self.children[variable]:
                    try:
                        assert variable in parents[variable2]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable} not in parents of {variable2}")
                    try:
                        assert self.timesteps[variable2] > self.timesteps[variable]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable2} has an earlier timestep than {variable}")
        
        # Sort variables by timestep
        self.variables.sort(key=lambda x: self.timesteps[x])

        # Set positions for plotting
        self.print_pos = print_pos
        width = {_: 0 for _ in range(len(self.variables))}
        if self.print_pos is None:
            self.print_pos = dict()
        if "raw_input" not in self.print_pos:
            self.print_pos["raw_input"] =  (0, -2)
        for var in self.variables:
            if var not in self.print_pos:
                self.print_pos[var] = (width[self.timesteps[var]], self.timesteps[var])
                width[self.timesteps[var]] += 1

        # Initializing the equivalence classes of children values
        # that produce a given parent value is expensive
        self.equiv_classes = {}

    # FUNCTIONS FOR RUNNING THE MODEL

    def run_forward(self, intervention=None):
        """
        Run the causal model forward with optional interventions.
        
        Parameters:
        -----------
        intervention : dict, optional
            A dictionary mapping variables to their intervened values (default is None).
            
        Returns:
        --------
        dict
            A dictionary mapping each variable to its computed value.
        """
        total_setting = defaultdict(None)
        length = len(list(total_setting.keys()))
        while length != len(self.variables):
            for variable in self.variables:
                for variable2 in self.parents[variable]:
                    if variable2 not in total_setting:
                        continue
                if intervention is not None and variable in intervention:
                    total_setting[variable] = intervention[variable]
                else:
                    total_setting[variable] = self.mechanisms[variable](
                        *[total_setting[parent] for parent in self.parents[variable]]
                    )
            length = len(list(total_setting.keys()))
        return total_setting

    def run_interchange(self, input_setting, counterfactual_inputs):
        """
        Run the model with interchange interventions.
        """ 