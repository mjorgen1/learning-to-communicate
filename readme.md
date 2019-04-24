
# Hierarchical Multi-Agent Deep Reinforcement Learning to Develop Long-Term Coordination

Marie Ossenkopf, Mackenzie Jorgensen, & Kurt Geihs

<p align="center">
</p>

Multi-agent systems need to communicate to coordinate a shared task. We show that a recurrent neural network (RNN) can learn
a communication protocol for coordination, even if the actions to coordinate lie outside of the communication range. We also show that a single RNN is unable to do this if there is an independent action sequence necessary before the coordinated action can be executed. We propose a hierarchical deep reinforcement learning model for multi-agent systems that separates the communication and coordination task from the action picking through a hierarchical policy. As a testbed, we propose the Dungeon Lever Game and we extend the Differentiable Inter-Agent Learning (DIAL) framework from Foerster et al's work "Learning to Communicate with Deep Multi-Agent Reinforcement Learning" (https://arxiv.org/abs/1605.06676). 

We owe a great deal to Foerster et al.'s work. We extended their code here to solve another problem with multi-agent communication and went a step further by adding hierarchical learning. 

## Links

\- [ResearchGate pre-print](https://www.researchgate.net/publication/331200217_Hierarchical_Multi-Agent_Deep_Reinforcement_Learning_to_Develop_Long-Term_Coordination)

## Execution
```
$ # Requirements: nvidia-docker (v1)
$ # Build docker instance (takes a while)
$ ./build.sh
$ # Run docker instance
$ ./run.sh
$ # Run experiment e.g.
$ run_hierarchyplan_2-dial.sh
```

## ACM Reference Format
   Marie Ossenkopf, Mackenzie Jorgensen, and Kurt Geihs. 2019. Hierarchical Multi-Agent Deep Reinforcement Learning to Develop Long-Term Coordination. In The 34th ACM/SIGAPP Symposium on Applied Computing (SAC’19), April 8–12, 2019, Limassol, Cyprus. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3297280.3297371


## License (from Foerster's work)

Code licensed under the Apache License v2.0
