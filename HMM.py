

import random
import argparse
import codecs
import os
import warnings

import numpy
import pandas as pd

warnings.filterwarnings("ignore")
# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        with open(f"{basename}.trans", 'r') as trans_file:
            for line in trans_file:
                from_state, to_state, prob = line.split()
                if from_state not in self.transitions:
                    self.transitions[from_state] = {}
                self.transitions[from_state][to_state] = float(prob)

        with open(f"{basename}.emit", 'r') as emit_file:
            for line in emit_file:
                state, symbol, prob = line.split()
                if state not in self.emissions:
                    self.emissions[state] = {}
                self.emissions[state][symbol] = float(prob)



   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        observations = []
        states = []
        current_state = '#'
        for i in range(n):
            next_state = random.choices(
                list(self.transitions[current_state].keys()),
                weights=self.transitions[current_state].values()
            )[0]

            symbol = random.choices(
                list(self.emissions[next_state].keys()),
                weights=self.emissions[next_state].values()
            )[0]

            observations.append(symbol)
            current_state = next_state
            states.append(current_state)

        print(observations)
        return observations


    def forward(self, observation):
        with (open(f"{observation}", 'r') as obs_file):
            for line in obs_file:
                if len(line) > 1 and line.endswith("\n"):
                    obs = line.split()
                    obs.insert(0, '#')
                    t = len(obs)
                    arr = pd.DataFrame(0.0, index=list(self.transitions.keys()), columns=obs)
                    arr.loc['#'][0] = 1.

                    # Then we propagate forward.
                    # For every subsequent timestep, for each state, we multiply the probability of reaching that state from
                    # any prior state by the probability of seeing this observation given that state.
                    for i in range(1, t):
                        for s in self.transitions:
                            total_sum = 0.0
                            if s != '#' and obs[i] in self.emissions[s]:
                                for s2 in self.transitions:
                                    total_sum += arr.loc[s2][i-1] * self.transitions[s2][s] * self.emissions[s][obs[i]]
                                arr.loc[s][obs[i]] = total_sum

                    print(arr)


## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        with (open(f"{observation}", 'r') as obs_file):
            for line in obs_file:
                if len(line) > 1 and line.endswith("\n"):
                    obs = line.split()
                    obs.insert(0, '#')
                    t = len(obs)
                    arr = pd.DataFrame(0.0, index=list(self.transitions.keys()), columns=obs)
                    backpointer_tab = pd.DataFrame(index=list(self.transitions.keys()), columns=obs)
                    arr.loc['#'][0] = 1.

                    for i in range(1, t):
                        for s in self.transitions:
                            total_sum = 0.0
                            index = ''
                            if s != '#' and obs[i] in self.emissions[s]:
                                for s2 in self.transitions:
                                    val = arr.loc[s2][i-1] * self.transitions[s2][s] * self.emissions[s][obs[i]]
                                    if total_sum < val:
                                        total_sum = val
                                        index = s2
                                arr.loc[s][obs[i]] = total_sum
                                backpointer_tab.loc[s][obs[i]] = index

                    best_path = [arr.iloc[:, -1].idxmax()]
                    for i in range(t - 1, 1, -1):
                        best_path.insert(0, backpointer_tab.iloc[arr.index.get_loc(best_path[0]), i])

                    print(best_path)


hmm1 = HMM()
hmm1.load("partofspeech.browntags.trained")
hmm1.generate(20)
hmm1.forward("ambiguous_sents.obs")
hmm1.viterbi("ambiguous_sents.obs")
