#!/bin/bash
th commWith2States.lua \
-game WaitingPlan \
-game_nagents 2 \
-game_action_space 2 \
-game_comm_limited 1 \
-game_comm_bits 3 \
-game_comm_sigma 3 \
-nsteps 8 \
-gamma 0.6 \
-model_dial 1 \
-model_bn 1 \
-model_know_share 1 \
-model_action_aware 1 \
-model_rnn_size 512 \
-model_rnn_layers 1 \
-model_rnn 'lstm' \
-bs 10 \
-learningra
te 0.0005 \
-nepisodes 5000 \
-step 100 \
-step_test 10 \
-step_target 100 \
-eps 0.05 \
-model_dropout 0.5 \
-imitation_Learning 0 \
-cuda 1
