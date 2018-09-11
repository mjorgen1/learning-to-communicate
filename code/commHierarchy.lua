--[[

    Learning to Communicate with Deep Multi-Agent Reinforcement Learning

    @article{foerster2016learning,
        title={Learning to Communicate with Deep Multi-Agent Reinforcement Learning},
        author={Foerster, Jakob N and Assael, Yannis M and de Freitas, Nando and Whiteson, Shimon},
        journal={arXiv preprint arXiv:1605.06676},
        year={2016}
    }

]] --


-- Configuration
cmd = torch.CmdLine()
cmd:text()
cmd:text('Learning to Communicate with Deep Multi-Agent Reinforcement Learning')
cmd:text()
cmd:text('Options')

-- general options:
cmd:option('-seed', -1, 'initial random seed')
cmd:option('-threads', 1, 'number of threads')

-- gpu
cmd:option('-cuda', 0, 'cuda')

-- rl
cmd:option('-gamma', 1, 'discount factor')
cmd:option('-upper_eps', 0.05, 'epsilon-greedy policy')
cmd:option('-lower_eps', 0.05, 'epsilon-greedy policy')

-- model
cmd:option('-model_rnn', 'gru', 'rnn type')
cmd:option('-model_dial', 0, 'use dial connection or rial')
cmd:option('-model_comm_narrow', 1, 'combines comm bits')
cmd:option('-model_know_share', 1, 'knowledge sharing')
cmd:option('-model_action_aware', 1, 'last action used as input')
cmd:option('-model_upper_rnn_size', 128, 'rnn size')
cmd:option('-model_upper_rnn_layers', 2, 'rnn layers')
cmd:option('-model_lower_rnn_size', 128, 'rnn size')
cmd:option('-model_lower_rnn_layers', 2, 'rnn layers')
cmd:option('-upper_model_dropout', 0, 'dropout')
cmd:option('-lower_model_dropout', 0, 'dropout')
cmd:option('-model_bn', 1, 'batch normalisation')
cmd:option('-model_target', 1, 'use a target network')
cmd:option('-model_avg_q', 1, 'avearge q functions')

-- training
cmd:option('-bs', 32, 'batch size')
cmd:option('-learningrate', 5e-4, 'learningrate')
cmd:option('-nepisodes', 1e+6, 'number of episodes')
cmd:option('-nsteps', 10, 'number of steps')

cmd:option('-step', 1000, 'print every episodes')
cmd:option('-step_test', 10, 'print every episodes')
cmd:option('-step_target', 100, 'target network updates')

cmd:option('-filename', '', '')

-- games
-- ColorDigit
cmd:option('-game', 'ColorDigit', 'game name')
cmd:option('-game_dim', 28, '')
cmd:option('-game_bias', 0, '')
cmd:option('-game_colors', 2, '')
cmd:option('-game_use_mnist', 1, '')
cmd:option('-game_use_digits', 0, '')
cmd:option('-game_nagents', 2, '')
cmd:option('-game_upper_action_space', 2, '')
cmd:option('-game_lower_action_space', 2, '')
cmd:option('-game_comm_limited', 0, '')
cmd:option('-game_comm_bits', 1, '')
cmd:option('-game_comm_sigma', 0, '')
cmd:option('-game_coop', 1, '')
cmd:option('-game_bottleneck', 10, '')
cmd:option('-game_level', 'extra_hard', '')
cmd:option('-game_vision_net', 'mlp', 'mlp or cnn')
cmd:option('-nsteps', 2, 'number of steps')
-- Switch
cmd:option('-game', 'Switch', 'game name')
cmd:option('-game_nagents', 3, '')
cmd:option('-game_upper_action_space', 2, '')
cmd:option('-game_lower_action_space', 2, '')
cmd:option('-game_comm_limited', 1, '')
cmd:option('-game_comm_bits', 2, '')
cmd:option('-game_comm_sigma', 0, '')
cmd:option('-nsteps', 6, 'number of steps')
-- SimplePlan
cmd:option('-imitation_Learning', 0, 'imitation learning')

cmd:text()

local opt = cmd:parse(arg)

-- Custom options
if opt.seed == -1 then opt.seed = torch.random(1000000) end
opt.model_comm_narrow = opt.model_dial

if opt.model_rnn == 'lstm' then
    opt.model_upper_rnn_states = 2 * opt.model_upper_rnn_layers
    opt.model_lower_rnn_states = 2 * opt.model_lower_rnn_layers
elseif opt.model_rnn == 'gru' then
    opt.model_upper_rnn_states = opt.model_upper_rnn_layers
    opt.model_lower_rnn_states = opt.model_lower_rnn_layers
end

-- Requirements
require 'nn'
require 'optim'
local kwargs = require 'include.kwargs'
local log = require 'include.log'
local util = require 'include.util'

-- Set float as default type
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

-- Cuda initialisation
if opt.cuda == 1 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(1)
    opt.dtype = 'torch.CudaTensor'
    print(cutorch.getDeviceProperties(1))
else
    opt.dtype = 'torch.FloatTensor'
end

if opt.model_comm_narrow == 0 and opt.game_comm_bits > 0 then
    opt.game_comm_bits = 2 ^ opt.game_comm_bits
end

-- Initialise game
local game = (require('game.' .. opt.game))(opt)

if opt.game_comm_bits > 0 and opt.game_nagents > 1 then
    -- Without dial we add the communication actions to the action space
    opt.game_upper_action_space_total = opt.game_upper_action_space + opt.game_comm_bits
else
    opt.game_upper_action_space_total = opt.game_upper_action_space
end

opt.game_lower_action_space_total = opt.game_lower_action_space

-- Initialise models
local upperModel = (require('model.' .. 'HierarchyUpperPlan'))(opt)
local lowerModel = (require('model.' .. 'HierarchyLowerPlan'))(opt)

-- Print options
util.sprint(opt)

-- Model target evaluate
upperModel.evaluate(upperModel.agent_target)
lowerModel.evaluate(lowerModel.agent_target)

-- Get parameters
local upper_params, upper_gradParams, upper_params_target, _ = upperModel.getParameters()
local lower_params, lower_gradParams, lower_params_target, _ = lowerModel.getParameters()

-- Optimisation function
local upper_optim_func, upper_optim_config = upperModel.optim()
local lower_optim_func, lower_optim_config = lowerModel.optim()
local optim_state = {}

local upper_learningrate = opt.learningrate
local lower_learningrate = opt.learningrate

-- Initialise agents
local agent = {}
for i = 1, opt.game_nagents do
    agent[i] = {}
print('initialised agent')
    agent[i].id = torch.Tensor():type(opt.dtype):resize(opt.bs):fill(i)

	--upper
    -- Populate init state
    agent[i].upper_input = {}
    agent[i].upper_state = {}
    agent[i].upper_state[0] = {}
    for j = 1, opt.model_upper_rnn_states do
        agent[i].upper_state[0][j] = torch.zeros(opt.bs, opt.model_upper_rnn_size):type(opt.dtype)
    end

    agent[i].upper_d_state = {}
    agent[i].upper_d_state[0] = {}
    for j = 1, opt.model_upper_rnn_states do
        agent[i].upper_d_state[0][j] = torch.zeros(opt.bs, opt.model_upper_rnn_size):type(opt.dtype)
    end


	--lower
    -- Populate init state
    agent[i].lower_input = {}
    agent[i].lower_input_target = {}
    agent[i].lower_state = {}
    agent[i].lower_state_target = {}
    agent[i].lower_state[0] = {}
    agent[i].lower_state_target[0] = {}
    for j = 1, opt.model_lower_rnn_states do
        agent[i].lower_state[0][j] = torch.zeros(opt.bs, opt.model_lower_rnn_size):type(opt.dtype)
        agent[i].lower_state_target[0][j] = torch.zeros(opt.bs, opt.model_lower_rnn_size):type(opt.dtype)
    end

    agent[i].lower_d_state = {}
    agent[i].lower_d_state[0] = {}
    for j = 1, opt.model_lower_rnn_states do
        agent[i].lower_d_state[0][j] = torch.zeros(opt.bs, opt.model_lower_rnn_size):type(opt.dtype)
    end

    -- Store q values
    agent[i].q_next_max = {}
end
print('populated agent')

local episode = {}

-- Initialise aux vectors
local upper_d_err = torch.Tensor(opt.bs, opt.game_upper_action_space_total):type(opt.dtype)
local upper_td_err = torch.Tensor(opt.bs):type(opt.dtype)
local upper_td_comm_err = torch.Tensor(opt.bs):type(opt.dtype)
local lower_d_err = torch.Tensor(opt.bs, opt.game_lower_action_space_total):type(opt.dtype)
local lower_td_err = torch.Tensor(opt.bs):type(opt.dtype)
local stats = {
    r_episode = torch.zeros(opt.nsteps),
    upper_td_err = torch.zeros(opt.step),
    upper_td_comm = torch.zeros(opt.step),
    lower_td_err = torch.zeros(opt.step),
    upper_train_r = torch.zeros(opt.step, opt.game_nagents),
    lower_train_r = torch.zeros(opt.step, opt.game_nagents),
    steps = torch.zeros(opt.step / opt.step_test),
    upper_test_r = torch.zeros(opt.step / opt.step_test, opt.game_nagents),
    lower_test_r = torch.zeros(opt.step / opt.step_test, opt.game_nagents),
    comm_per = torch.zeros(opt.step / opt.step_test),
    te = torch.zeros(opt.step)
}

local replay = {}
print('initialised aux vectors')
-- Run episode






local function run_episode(opt, game, upperModel, lowerModel, agent, e, test_mode)
--print('run episode '.. e) 
    -- Test mode
    test_mode = test_mode or false

    -- Reset game
    game:reset(e)
    local im_learning = false
    --allow imitation learning every even numbered episodes
    if(opt.imitation_Learning == 1) then
        if(e % 2 == 0  ) then
            im_learning = true
        else
            im_learning = false
        end
    end

    -- Initialise episode
    local step = 1
    local upper_episode = {
        r = torch.zeros(opt.bs, opt.game_nagents),
    }   
    local lower_episode = {
        r = torch.zeros(opt.bs, opt.game_nagents),
        steps = torch.zeros(opt.bs),
        ended = torch.zeros(opt.bs),
    }
    upper_episode[step] = {
        s_t = game:getState(),
        terminal = torch.zeros(opt.bs)
    }
    lower_episode[step] = {
        s_t = game:getState(),
        terminal = torch.zeros(opt.bs)
    }
    if opt.game_comm_bits > 0 and opt.game_nagents > 1 then
        upper_episode[step].comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits):type(opt.dtype)
        if opt.model_dial == 1 and opt.model_target == 1 then
            upper_episode[step].comm_target = upper_episode[step].comm:clone()
        end
        upper_episode[step].d_comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits):type(opt.dtype)
    end

    --Run upper level for 2 steps
    local steps = 2
    while step <= steps do

        -- Initialise next step
        upper_episode[step + 1] = {}

        -- Initialise comm channel
        if opt.game_comm_bits > 0 and opt.game_nagents > 1 then
            upper_episode[step + 1].comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits):type(opt.dtype)
            upper_episode[step + 1].d_comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits):type(opt.dtype)
            if opt.model_dial == 1 and opt.model_target == 1 then
                upper_episode[step + 1].comm_target = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits):type(opt.dtype)
            end
        end

        -- Forward pass
        upper_episode[step].a_t = torch.zeros(opt.bs, opt.game_nagents):type(opt.dtype)
        if opt.model_dial == 0 then
            upper_episode[step].a_comm_t = torch.zeros(opt.bs, opt.game_nagents):type(opt.dtype)
        end

        -- Iterate agents
        for i = 1, opt.game_nagents do
            agent[i].upper_input[step] = {
                upper_episode[step].s_t[i][{{},{1}}]:squeeze():type(opt.dtype),
                agent[i].id,
                agent[i].upper_state[step - 1]
            }


            -- Communication enabled
            if opt.game_comm_bits > 0 and opt.game_nagents > 1 then
                local comm = upper_episode[step].comm:clone():type(opt.dtype)
                -- Create limited communication channel nbits
                local comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits):type(opt.dtype)
                for b = 1, opt.bs do
		    if i == 1 then
                    	comm_lim[{ { b } }] = comm[{ { b }, unpack({ 2, {} }) }]
                    elseif i == 2 then
                        comm_lim[{ { b } }] = comm[{ { b }, unpack({ 1, {} }) }]
		    end
                end

                table.insert(agent[i].upper_input[step], comm_lim)

		if test_mode then
                  --  print("\n")
                   -- print("The comm sent to agent".. i)
	           -- print(comm_lim[1])

		end
            end

            -- Compute Q values
            local comm, state, q_t
            agent[i].upper_state[step], q_t = unpack(upperModel.agent[upperModel.id(step, i)]:forward(agent[i].upper_input[step]))


            -- If dial split out the comm values from q values
            if opt.model_dial == 1 then
                q_t, comm = DRU(q_t, test_mode)
            end

            --Print the communication for each agent
            if test_mode then
              --  print("Agent " .. i .. "'s Current state: "  .. upper_episode[step].s_t[i][1][1])
              --  print(q_t[1]:view(1,-1))
            end

            -- Pick an action (epsilon-greedy)
            local action_range, action_range_comm
            local max_value, max_a, max_a_comm
            if opt.model_dial == 0 then
                action_range, action_range_comm = game:getUpperActionRange(step, i)
            else
                action_range = game:getUpperActionRange(step, i)
            end

            -- If Limited action range
            if action_range then
                agent[i].upper_range = agent[i].upper_range or torch.range(1, opt.game_upper_action_space_total)
                max_value = torch.Tensor(opt.bs, 1)
                max_a = torch.zeros(opt.bs, 1)
                if opt.model_dial == 0 then
                    max_a_comm = torch.zeros(opt.bs, 1)
                end
                for b = 1, opt.bs do
                    -- If comm always fetch range for comm and actions
                    if opt.model_dial == 0 then
                        -- If action was taken
                        if action_range[b][2][1] > 0 then
                            local v, a = torch.max(q_t[action_range[b]], 2)
                            max_value[b] = v:squeeze()
                            max_a[b] = agent[i].upper_range[{ action_range[b][2] }][a:squeeze()]
                        end
                        -- If comm action was taken
                        if action_range_comm[b][2][1] > 0 then
                            local v, a = torch.max(q_t[action_range_comm[b]], 2)
                            max_a_comm[b] = agent[i].upper_range[{ action_range_comm[b][2] }][a:squeeze()]
                        end
                    else
                        local v, a = torch.max(q_t[action_range[b]], 2)
                        max_a[b] = agent[i].upper_range[{ action_range[b][2] }][a:squeeze()]
                    end
                end
            else
                -- If comm always pick max_a and max_comm
                if opt.model_dial == 0 and opt.game_comm_bits > 0 then
                    _, max_a = torch.max(q_t[{ {}, { 1, opt.game_upper_action_space } }], 2)
                    _, max_a_comm = torch.max(q_t[{ {}, { opt.game_upper_action_space + 1, opt.game_upper_action_space_total } }], 2)
                    max_a_comm = max_a_comm + opt.game_upper_action_space
                else
                    _, max_a = torch.max(q_t, 2)
                end
            end

            -- Store actions
            upper_episode[step].a_t[{ {}, { i } }] = max_a:type(opt.dtype)
            if test_mode then --prints out the actions for the test mode
               -- print("The action for agent " .. i .. " is ")
               -- print(upper_episode[step].a_t[1][i])
            end
            if opt.model_dial == 0 and opt.game_comm_bits > 0 then
                upper_episode[step].a_comm_t[{ {}, { i } }] = max_a_comm:type(opt.dtype)
            end
            for b = 1, opt.bs do

                -- Epsilon-greedy action picking
                if not test_mode then
                    if opt.model_dial == 0 then
                        -- Random action
                        if torch.uniform() < opt.upper_eps then

                            if action_range then
                                if action_range[b][2][1] > 0 then
                                    local a_range = agent[i].upper_range[{ action_range[b][2] }]
                                    local a_idx = torch.random(a_range:nElement())
                                    upper_episode[step].a_t[b][i] = agent[i].upper_range[{ action_range[b][2] }][a_idx]
                                end
                            else
                                upper_episode[step].a_t[b][i] = torch.random(opt.game_upper_action_space)
                            end
                        end

                        -- Random communication
                        if opt.game_comm_bits > 0 and torch.uniform() < opt.upper_eps then
                            if action_range then
                                if action_range_comm[b][2][1] > 0 then
                                    local a_range = agent[i].upper_range[{ action_range_comm[b][2] }]
                                    local a_idx = torch.random(a_range:nElement())
                                    upper_episode[step].a_comm_t[b][i] = agent[i].upper_range[{ action_range_comm[b][2] }][a_idx]
                                end
                            else
                                upper_episode[step].a_comm_t[b][i] = torch.random(opt.game_upper_action_space + 1, opt.game_upper_action_space_total)
                            end
                        end

                    else
                        if torch.uniform() < opt.upper_eps then
                            if action_range then
                                local a_range = agent[i].upper_range[{ action_range[b][2] }]
                                local a_idx = torch.random(a_range:nElement())
                                upper_episode[step].a_t[b][i] = agent[i].upper_range[{ action_range[b][2] }][a_idx]
                                --if b == 1 then --if on first batch for test phase print out actions for the agents
                                --    print("When testmode " .. (test_mode and 'true' or 'false') .. " the action for agent " .. i .. " is ")
                                --    print(episode[step].a_t[b][i])
                                --end
                            else
                                upper_episode[step].a_t[b][i] = torch.random(q_t[b]:size(1))
                            end
                        end
                    end
                end

                -- If communication action populate channel
                if step <= opt.nsteps then
                    -- For dial we 'forward' the direct activation otherwise we shift the a_t into the 1-game_comm_bits range
                    if opt.model_dial == 1 then
                        upper_episode[step + 1].comm[b][i] = comm[b]
                    else
                        local a_t = upper_episode[step].a_comm_t[b][i] - opt.game_upper_action_space
                        if a_t > 0 then
                            upper_episode[step + 1].comm[b][{ { i }, { a_t } }] = 1
                        end
                    end


                else

                end
            end

	end

        -- Forward next step
        step = step + 1
        upper_episode[step].s_t = game:getState()

    end

    upper_episode[1].r_t=torch.zeros(opt.bs, opt.game_nagents)

    time_target = upper_episode[2].a_t

    --set back for lower policy
    step = 1

    -- Run for N steps
    local steps = test_mode and opt.nsteps or opt.nsteps + 1
    while step <= steps and lower_episode.ended:sum() < opt.bs do

        -- Initialise next step
        lower_episode[step + 1] = {}

        -- Forward pass
        lower_episode[step].a_t = torch.zeros(opt.bs, opt.game_nagents):type(opt.dtype)

        -- Iterate agents
        for i = 1, opt.game_nagents do
            agent[i].lower_input[step] = {
		time_target[{{},{i}}]:squeeze():type(opt.dtype),
                lower_episode[step].s_t[i][{{},{1}}]:squeeze():type(opt.dtype),
                lower_episode[step].s_t[i][{{},{2}}]:squeeze():type(opt.dtype),
                agent[i].id,
                agent[i].lower_state[step - 1]
            }


            -- Last action enabled
            if opt.model_action_aware == 1 then
                -- If comm always then use both action
                -- Action aware for single a, comm action
                local la = torch.ones(opt.bs):type(opt.dtype)
                if step > 1 then
                    for b = 1, opt.bs do
                        if lower_episode[step - 1].a_t[b][i] > 0 then
                            la[{ { b } }] = lower_episode[step - 1].a_t[b][i] + 1
                        end
                    end
                end
                table.insert(agent[i].lower_input[step], la)
            end
            -- Compute Q values
            local comm, state, q_t
            agent[i].lower_state[step], q_t = unpack(lowerModel.agent[lowerModel.id(step, i)]:forward(agent[i].lower_input[step]))


            --Print the communication for each agent
            if test_mode then
              --  print("Agent " .. i .. "'s Current state: " .. time_target[1][i] .. ' '.. lower_episode[step].s_t[i][1][1] .. ' ' .. lower_episode[step].s_t[i][1][2] )
              --  print(q_t[1]:view(1,-1))
            end

            -- Pick an action (epsilon-greedy)
            local action_range, action_range_comm
            local max_value, max_a
            action_range = game:getLowerActionRange(step, i)

            -- If Limited action range
            if action_range then
                agent[i].lower_range = agent[i].lower_range or torch.range(1, opt.game_lower_action_space_total)
                max_value = torch.Tensor(opt.bs, 1)
                max_a = torch.zeros(opt.bs, 1)
                for b = 1, opt.bs do
                    -- If comm always fetch range for comm and actions
                    local v, a = torch.max(q_t[action_range[b]], 2)
                    max_a[b] = agent[i].lower_range[{ action_range[b][2] }][a:squeeze()]
                end
            else
                -- If comm always pick max_a and max_comm
                _, max_a = torch.max(q_t, 2)
            end

            -- Store actions
            lower_episode[step].a_t[{ {}, { i } }] = max_a:type(opt.dtype)
            if test_mode then --prints out the actions for the test mode
              --  print("The action for agent " .. i .. " is ")
              --  print(lower_episode[step].a_t[1][i])
            end

	    --asking for adviced action from game
            local adviced_actions = game:imitateAction()


            for b = 1, opt.bs do

                -- Epsilon-greedy action picking
                if not test_mode and not im_learning then
                    if torch.uniform() < opt.lower_eps then
                        if action_range then
                            local a_range = agent[i].lower_range[{ action_range[b][2] }]
                            local a_idx = torch.random(a_range:nElement())
                            lower_episode[step].a_t[b][i] = agent[i].lower_range[{ action_range[b][2] }][a_idx]
                        else
                            lower_episode[step].a_t[b][i] = torch.random(q_t[b]:size(1))
                        end
                    end
                end
		-- use adviced action for imitation learning
                if not test_mode and im_learning then
                    lower_episode[step].a_t[b][i] = adviced_actions[b][i]
                end
            end
        end

        -- Compute reward for current state-action pair
        lower_episode[step].r_t, lower_episode[step].terminal = game:lower_step(lower_episode[step].a_t,time_target)
	
	if test_mode then
	  --  print('reward achieved: ')
	  --  print(lower_episode[step].r_t[1])
	  --  print('terminated: '.. lower_episode[step].terminal[1])
	end

        -- Accumulate steps (not for +1 step)
        if step <= opt.nsteps then
            for b = 1, opt.bs do
                if lower_episode.ended[b] == 0 then

                    -- Keep steps and rewards
                    lower_episode.steps[{ { b } }]:add(1)
                    lower_episode.r[{ { b } }]:add(lower_episode[step].r_t[b])

                    -- Check if terminal
                    if lower_episode[step].terminal[b] == 1 then
                        lower_episode.ended[{ { b } }] = 1
                    end
                end
            end
        end


        -- Target Network, for look-ahead
        if opt.model_target == 1 and not test_mode then
            for i = 1, opt.game_nagents do

                -- Target input
                agent[i].lower_input_target[step] = {
                    agent[i].lower_input[step][1],
                    agent[i].lower_input[step][2],
                    agent[i].lower_input[step][3],
                    agent[i].lower_input[step][4],
                    agent[i].lower_state_target[step - 1],
                    agent[i].lower_input[step][6],
                }

                -- Forward target
                local state, q_t_target = unpack(lowerModel.agent_target[lowerModel.id(step, i)]:forward(agent[i].lower_input_target[step]))
                agent[i].lower_state_target[step] = state

                -- Limit actions
                local action_range = game:getLowerActionRange(step, i)
                if action_range then
                    agent[i].q_next_max[step] = torch.zeros(opt.bs):type(opt.dtype)
                    for b = 1, opt.bs do
                        if action_range[b][2][1] > 0 then
                            agent[i].q_next_max[step][b], _ = torch.max(q_t_target[action_range[b]], 2)
                        end
                    end
                else
                    agent[i].q_next_max[step], _ = torch.max(q_t_target, 2)
                end
            end
        end

        -- Forward next step
        step = step + 1
        if lower_episode.ended:sum() < opt.bs then
            lower_episode[step].s_t = game:getState()
        end
    end

    upper_episode[2].r_t = game:upper_step()
    upper_episode.r = upper_episode[2].r_t
    upper_episode.steps = 2* torch.ones(opt.bs)

    lower_episode.nsteps = lower_episode.steps:max()
    upper_episode.nsteps = upper_episode.steps:max()

    return upper_episode, lower_episode, agent
end





-- split out the communication bits and add noise.
function DRU(q_t, test_mode)
    if opt.model_dial == 0 then error('Warning!! Should only be used in DIAL') end
    local bound = opt.game_upper_action_space
    local q_t_n = q_t[{ {}, { 1, bound } }]:clone()
    local comm = q_t[{ {}, { bound + 1, opt.game_upper_action_space_total } }]:clone()
    if test_mode then
        if opt.model_comm_narrow == 0 then
            local ind
            _, ind = torch.max(comm, 2)
            comm:zero()
            for b = 1, opt.bs do
                comm[b][ind[b][1]] = 20
            end
        else
            comm = comm:gt(0.5):type(opt.dtype):add(-0.5):mul(2 * 20)
        end
    end
    if opt.game_comm_sigma > 0 and not test_mode then
        local noise_vect = torch.randn(comm:size()):type(opt.dtype):mul(opt.game_comm_sigma)
        comm = comm + noise_vect
    end
    return q_t_n, comm
end






-- Start time
local beginning_time = torch.tic()
print('start time')
-- Iterate episodes
for e = 1, opt.nepisodes do

    stats.e = e

    -- Initialise clock
    local time = sys.clock()

    -- Model training
    upperModel.training(upperModel.agent)
    lowerModel.training(lowerModel.agent)

    --Print which epoch the run is on
    --print("Current epoch: " .. e)

    -- Run episode
    upper_episode , lower_episode , agent = run_episode(opt, game, upperModel, lowerModel, agent, e)

    -- Rewards stats
    stats.upper_train_r[(e - 1) % opt.step + 1] = upper_episode.r:mean(1)
    stats.lower_train_r[(e - 1) % opt.step + 1] = lower_episode.r:mean(1)

    -- Reset parameters
    if e == 1 then
        upper_gradParams:zero()
        lower_gradParams:zero()
    end

    -- Lower Backward pass
    local step_back = 1
    for step = lower_episode.nsteps, 1, -1 do --iterates backwards through the steps
        stats.lower_td_err[(e - 1) % opt.step + 1] = 0

        -- Iterate agents
        for i = 1, opt.game_nagents do

            -- Compute Q values
            local state, q_t = unpack(lowerModel.agent[lowerModel.id(step, i)].output)

            -- Compute td error
            lower_td_err:zero()
            lower_d_err:zero()

            for b = 1, opt.bs do
                if step >= lower_episode.steps[b] then
                    -- if first backward init RNN
                    for j = 1, opt.model_lower_rnn_states do
                        agent[i].lower_d_state[step_back - 1][j][b]:zero()
                    end
                end

                if step <= lower_episode.steps[b] then

                    -- if terminal state or end state => no future rewards
                    if lower_episode[step].a_t[b][i] > 0 then
                        if lower_episode[step].terminal[b] == 1 then
                            lower_td_err[b] = lower_episode[step].r_t[b][i] - q_t[b][lower_episode[step].a_t[b][i]]
                        else
                            local q_next_max
                            q_next_max = agent[i].q_next_max[step + 1]:squeeze()
                            lower_td_err[b] = lower_episode[step].r_t[b][i] + opt.gamma * q_next_max[b] - q_t[b][lower_episode[step].a_t[b][i]]
                        end
                        lower_d_err[{ { b }, { lower_episode[step].a_t[b][i] } }] = -lower_td_err[b]

                    else
                        error('Error!')
                    end

                end
            end


            -- Track td-err
            stats.lower_td_err[(e - 1) % opt.step + 1] = stats.lower_td_err[(e - 1) % opt.step + 1] + 0.5 * lower_td_err:clone():pow(2):mean()

            -- Backward pass
            local lower_grad = lowerModel.agent[lowerModel.id(step, i)]:backward(agent[i].lower_input[step], {
                agent[i].lower_d_state[step_back - 1],
                lower_d_err
            })

            --'state' is the 3rd input, so we can extract d_state
            agent[i].lower_d_state[step_back] = lower_grad[5]

            --For dial we need to write add the derivatives w/ respect to the incoming messages to the d_comm tracker
        end

        -- Count backward steps
        step_back = step_back + 1

    end

    -- Upper Backward pass
    local step_back = 1
    for step = upper_episode.nsteps, 1, -1 do --iterates backwards through the steps
        stats.upper_td_err[(e - 1) % opt.step + 1] = 0
        stats.upper_td_comm[(e - 1) % opt.step + 1] = 0

        -- Iterate agents
        for i = 1, opt.game_nagents do

            -- Compute Q values
            local state, q_t = unpack(upperModel.agent[upperModel.id(step, i)].output)

            -- Compute td error
            upper_td_err:zero()
            upper_td_comm_err:zero()
            upper_d_err:zero()

            for b = 1, opt.bs do
                if step >= upper_episode.steps[b] then
                    -- if first backward init RNN
                    for j = 1, opt.model_upper_rnn_states do
                        agent[i].upper_d_state[step_back - 1][j][b]:zero()
                    end
                end

                if step <= upper_episode.steps[b] then

                    -- if terminal state or end state => no future rewards
                    if upper_episode[step].a_t[b][i] > 0 then
                        upper_td_err[b] = upper_episode[step].r_t[b][i] - q_t[b][upper_episode[step].a_t[b][i]]
                        upper_d_err[{ { b }, { upper_episode[step].a_t[b][i] } }] = -upper_td_err[b]
                    else
                        error('Error!')
                    end

                    -- If we use dial and the agent took the umbrella comm action and the messsage happened before last round, the we get incoming derivaties
                    if opt.model_dial == 1 and step < upper_episode.steps[b] then
                        -- Derivatives with respect to agent_i's message are stored in d_comm[b][i]
                        local bound = opt.game_upper_action_space
                        upper_d_err[{ { b }, { bound + 1, opt.game_upper_action_space_total } }]:add(upper_episode[step + 1].d_comm[b][i])
                    end
                end
            end


            -- Track td-err
            stats.upper_td_err[(e - 1) % opt.step + 1] = stats.upper_td_err[(e - 1) % opt.step + 1] + 0.5 * upper_td_err:clone():pow(2):mean()
            if opt.model_dial == 0 then
                stats.upper_td_comm[(e - 1) % opt.step + 1] = stats.upper_td_comm[(e - 1) % opt.step + 1] + 0.5 * upper_td_comm_err:clone():pow(2):mean()
            end

            -- Track the amplitude of the dial-derivatives
            if opt.model_dial == 1 then
                local bound = opt.game_upper_action_space
                stats.upper_td_comm[(e - 1) % opt.step + 1] = stats.upper_td_comm[(e - 1) % opt.step + 1] + 0.5 * upper_d_err[{ {}, { bound + 1, opt.game_upper_action_space_total } }]:clone():pow(2):mean()
            end

            -- Backward pass
            local upper_grad = upperModel.agent[upperModel.id(step, i)]:backward(agent[i].upper_input[step], {
                agent[i].upper_d_state[step_back - 1],
                upper_d_err
            })

            --'state' is the 3rd input, so we can extract d_state
            agent[i].upper_d_state[step_back] = upper_grad[3]

            --For dial we need to write add the derivatives w/ respect to the incoming messages to the d_comm tracker
            if opt.model_dial == 1 then
                local comm_grad = upper_grad[4]

                local comm_lim = {}
                for b = 1, opt.bs do
		    if i == 1 then
                    	comm_lim[b] = { 2, {} }
                    elseif i == 2 then
                        comm_lim[b] = { 1, {} }
		    end
                end

                for b = 1, opt.bs do
                    -- Agent could only receive the message if they were active
                    upper_episode[step].d_comm[{ { b }, unpack(comm_lim[b]) }]:add(comm_grad[b])
                end
            end
        end

        -- Count backward steps
        step_back = step_back + 1
    end

    -- Update gradients
    local upper_feval = function(x)

        -- Normalise Gradients
        upper_gradParams:div(opt.game_nagents * opt.bs)

        -- Clip Gradients
        upper_gradParams:clamp(-10, 10)

        return nil, upper_gradParams
    end

    -- Update gradients
    local lower_feval = function(x)

        -- Normalise Gradients
        lower_gradParams:div(opt.game_nagents * opt.bs)

        -- Clip Gradients
        lower_gradParams:clamp(-10, 10)

        return nil, lower_gradParams
    end

 
    upper_optim_config.learningRate = upper_learningrate
    upper_optim_func(upper_feval, upper_params, upper_optim_config, optim_state)

    lower_optim_config.learningRate = lower_learningrate
    lower_optim_func(lower_feval, lower_params, lower_optim_config, optim_state)

    -- Gradient statistics
    if e % opt.step == 0 then
        stats.upper_grad_norm = upper_gradParams:norm() / upper_gradParams:nElement() * 1000
        stats.lower_grad_norm = lower_gradParams:norm() / lower_gradParams:nElement() * 1000
    end


    -- Reset parameters
    upper_gradParams:zero()
    lower_gradParams:zero()

    -- Update target network
    if e % opt.step_target == 0 then
        lower_params_target:copy(lower_params)
    end

    -- Test
    if e % opt.step_test == 0 then
        local test_idx = (e / opt.step_test - 1) % (opt.step / opt.step_test) + 1

        local upper_episode , lower_episode , _ = run_episode(opt, game, upperModel, lowerModel, agent, e, true)
        stats.upper_test_r[test_idx] = upper_episode.r:mean(1)
        stats.lower_test_r[test_idx] = lower_episode.r:mean(1)
        stats.steps[test_idx] = lower_episode.steps:mean()

	te_r = lower_episode.r:mean(1):mean(2):type(opt.dtype):squeeze()
	
        if te_r >= 0.99 then
	    lower_learningrate = opt.learningrate *0.001
	    opt.lower_eps = 0.002
	    upper_learningrate = opt.learningrate
        elseif te_r >= 0.97 then
	    lower_learningrate = opt.learningrate *0.01
	    opt.lower_eps = 0.005
	    upper_learningRate = opt.learningrate *0.1
        elseif te_r >= 0.95 then
	    lower_learningrate = opt.learningrate *0.1
	    opt.lower_eps = 0.01
	    upper_learningRate = opt.learningrate *0.01
        elseif te_r <= 0.8 then
	    lower_learningrate = opt.learningrate
	    opt.lower_eps = 0.05
	    upper_learningRate = opt.learningrate * 0.001
	end
    end

    -- Compute statistics
    stats.te[(e - 1) % opt.step + 1] = sys.clock() - time

    if e == opt.step then
        stats.td_err_avg = stats.upper_td_err:mean()
        stats.td_comm_avg = stats.upper_td_comm:mean()
        stats.upper_train_r_avg = stats.upper_train_r:mean(1)
        stats.upper_test_r_avg = stats.upper_test_r:mean(1)
        stats.lower_train_r_avg = stats.lower_train_r:mean(1)
        stats.lower_test_r_avg = stats.lower_test_r:mean(1)
        stats.steps_avg = stats.steps:mean()
        stats.comm_per_avg = stats.comm_per:mean()
        stats.te_avg = stats.te:mean()
    elseif e % opt.step == 0 then
        local coef = 0.9
        stats.td_err_avg = stats.td_err_avg * coef + stats.upper_td_err:mean() * (1 - coef)
        stats.td_comm_avg = stats.td_comm_avg * coef + stats.upper_td_comm:mean() * (1 - coef)
        stats.upper_train_r_avg = stats.upper_train_r_avg * coef + stats.upper_train_r:mean(1) * (1 - coef)
        stats.upper_test_r_avg = stats.upper_test_r_avg * coef + stats.upper_test_r:mean(1) * (1 - coef)
        stats.lower_train_r_avg = stats.lower_train_r_avg * coef + stats.lower_train_r:mean(1) * (1 - coef)
        stats.lower_test_r_avg = stats.lower_test_r_avg * coef + stats.lower_test_r:mean(1) * (1 - coef)
        stats.steps_avg = stats.steps_avg * coef + stats.steps:mean() * (1 - coef)
        stats.comm_per_avg = stats.comm_per_avg * coef + stats.comm_per:mean() * (1 - coef)
        stats.te_avg = stats.te_avg * coef + stats.te:mean() * (1 - coef)
    end

    -- Print statistics
    if e % opt.step == 0 then
        log.infof('e=%d, td_err=%.3f, td_err_avg=%.3f, td_comm=%.3f, td_comm_avg=%.3f, up_tr_r=%.2f, up_tr_r_avg=%.2f, up_te_r=%.2f, up_te_r_avg=%.2f, low_tr_r=%.2f, low_tr_r_avg=%.2f, low_te_r=%.2f, low_te_r_avg=%.2f, st=%.1f, comm=%.1f%%, grad=%.3f, t/s=%.2f s, t=%d m',
            stats.e,
            stats.upper_td_err:mean(),
            stats.td_err_avg,
            stats.upper_td_comm:mean(),
            stats.td_comm_avg,
            stats.upper_train_r:mean(),
            stats.upper_train_r_avg:mean(),
            stats.upper_test_r:mean(),
            stats.upper_test_r_avg:mean(),
            stats.lower_train_r:mean(),
            stats.lower_train_r_avg:mean(),
            stats.lower_test_r:mean(),
            stats.lower_test_r_avg:mean(),
            stats.steps_avg,
            stats.comm_per_avg * 100,
            stats.upper_grad_norm,
            stats.te_avg * opt.step,
            torch.toc(beginning_time) / 60)
        collectgarbage()
    end

    -- run model specific statistics
    upperModel.stats(opt, game, stats, e)
    lowerModel.stats(opt, game, stats, e)
    -- run model specific statistics
    --upperModel.save(opt, stats, upperModel)
    --lowerModel.save(opt, stats, lowerModel)
end

