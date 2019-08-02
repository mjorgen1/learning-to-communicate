#!/bin/bash
th commUpperHierarchy.lua \
-game UpperHierarchyPlan \
-game_nagents 2 \
-game_upper_action_space 7 \
-game_lower_action_space 4 \
-game_comm_limited 1 \
-game_comm_bits 1 \
-game_comm_sigma 1 \
-nsteps 6 \
-gamma 0.7 \
-model_dial 1 \
-model_bn 1 \
-model_know_share 1 \
-model_action_aware 1 \
-decision_model_size 200 \
-model_lower_rnn_size 512 \
-model_lower_rnn_layers 1 \
-model_rnn 'lstm' \
-bs 32 \
-learningrate 0.0005 \
-nepisodes 31250 \
-step 100 \
-step_test 10 \
-step_target 100 \
-upper_eps 0.05 \
-lower_eps 0.05 \
-upper_model_dropout 0.3 \
-lower_model_dropout 0.2 \
-imitation_Learning 0 \
-cuda 1