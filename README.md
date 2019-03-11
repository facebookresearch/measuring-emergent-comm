# Measuring emergent communication

This repo contains the code used to run the experiments in our paper 
"On the Pitfalls of Measuring Emergent Communication", AAMAS 2019.
We include both code for training RL agents on matrix communication games (MCGs),
and code to evaluate their performance and calculate the causal influence of 
communication (CIC).

## Dependencies
- Python 3.6
- Pytorch >= 0.4.1
- Numpy  

## Get started

To train REINFORCE agents on random MCGs, simply run `main.py`. You can adjust the training parameters by 
passing the appropriate arguments (see the `parse_args()` function). By default, this will
print out each agent's reward and the speaker consistency (SC) value. 
You can view more outputs, including the instantaneous coordination (IC), and
 the action prediction accuracy, by using the option ``--verbose``. 

If you want to calculate the causal influence of communication, see the code provided
in `eval.py`, specifically the `calc_model_cic` function.

If you are just interested in using MCGs as an environment to train other agents, 
see the `mcg.py` file. 

## License
Measuring-emergent-comm is released under Creative Commons Attribution 4.0 International (CC BY 4.0) license, as found in the LICENSE file.

## Bibliography

If you found this code useful, please cite the following paper:

"On the Pitfalls of Measuring Emergent Communication", 
Ryan Lowe, Jakob Foerster, Y-Lan Boureau, Joelle Pineau, and Yann Dauphin, 
AAMAS 2019.

``
@inproceedings{lowe2019on,
  title = {On the pitfalls of measuring emergent communication},
  author = {Lowe, Ryan and Foerster, Jakob and Boureau, Y-Lan and Pineau, Joelle and Dauphin, Yann},
  booktitle = {International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year = {2019}
}
``