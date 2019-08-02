require 'nn'
require 'nngraph'
require 'optim'
require 'csvigo'

local kwargs = require 'include.kwargs'
local log = require 'include.log'
local util = require 'include.util'
local LSTM = require 'module.LSTM'
local GRU = require 'module.GRU'
require 'module.rmsprop'
require 'module.GaussianNoise'
require 'module.Binarize'

return function(opt)

    local exp = {}

    function exp.optim(iter)
        -- iter can be used for learning rate decay
        -- local optimfunc = optim.adam
        local optimfunc = optim.rmsprop
        local optimconfig = { learningRate = opt.learningrate }
        return optimfunc, optimconfig
    end

    function exp.save(opt, stats, model)
        if stats.e % opt.step == 0 then
            if opt.filename == '' then
                exp.save_path = exp.save_path or paths.concat('results', opt.game .. '_' .. opt.game_nagents ..
                        (opt.model_dial == 1 and '_dial' or '') .. '_' .. string.upper(string.format("%x", opt.seed)))
            else
                exp.save_path = exp.save_path or paths.concat('results', opt.game .. '_' .. opt.game_nagents ..
                        (opt.model_dial == 1 and '_dial' or '') .. '_' .. opt.filename .. '_' .. string.upper(string.format("%x", opt.seed)))
            end


            -- Save opt
            if stats.e == opt.step then
                os.execute('mkdir -p ' .. exp.save_path)
                local opt_csv = {}
                for k, v in util.spairs(opt) do
                    table.insert(opt_csv, { k, v })
                end

                csvigo.save({
                    path = paths.concat(exp.save_path, 'opt.csv'),
                    data = opt_csv,
                    verbose = false
                })
            end

            -- keep stats
            stats.history = stats.history or { { 'e', 'td_err', 'td_comm', 'train_r', 'test_r', 'test_opt', 'test_god', 'steps', 'comm_per', 'te' } }
            table.insert(stats.history, {
                stats.e,
                stats.td_err:mean(),
                stats.td_comm:mean(),
                stats.train_r:mean(),
                stats.test_r:mean(),
                stats.test_opt:mean(),
                stats.test_god:mean(),
                stats.steps:mean(),
                stats.comm_per:mean(),
                stats.te:mean()
            })

            -- Save stats csv
            csvigo.save({
                path = paths.concat(exp.save_path, 'stats.csv'),
                data = stats.history,
                verbose = false
            })

            -- Save action histogram
            if opt.hist_action == 1 then
                -- Append to memory
                stats.history_hist_action = stats.history_hist_action or {}
                table.insert(stats.history_hist_action,
                    stats.hist_action_avg:totable()[1])

                -- save csv
                csvigo.save({
                    path = paths.concat(exp.save_path, 'hist_action.csv'),
                    data = stats.history_hist_action,
                    verbose = false
                })
            end

            -- Save action histogram
            if opt.hist_comm == 1 then
                -- Append to memory
                stats.history_hist_comm = stats.history_hist_comm or {}
                table.insert(stats.history_hist_comm,
                    stats.hist_comm_avg:totable()[1])

                -- save csv
                csvigo.save({
                    path = paths.concat(exp.save_path, 'hist_comm.csv'),
                    data = stats.history_hist_comm,
                    verbose = false
                })
            end
            -- save model
            if stats.e % (opt.step * 10) == 0 then
                log.debug('Saving model')

                -- clear state
                -- exp.clearState(model.agent)

                -- save model
                local filename = paths.concat(exp.save_path, 'exp.t7')
                torch.save(filename, { opt, stats, model.agent })
            end
        end
    end

    function exp.load()
    end

    function exp.clearState(model)
        for i = 1, #model do
            model[i]:clearState()
        end
    end

    function exp.training(model)
        for i = 1, #model do
            model[i]:training()
        end
    end

    function exp.evaluate(model)
        for i = 1, #model do
            model[i]:evaluate()
        end
    end

    function exp.getParameters()
        -- Get model params
        local a = nn.Container()
        for i = 1, #exp.agent do
            a:add(exp.agent[i])
        end
        local params, gradParams = a:getParameters()

        log.infof('Creating model(s), params=%d', params:nElement())

        -- Get target model params
        local a = nn.Container()
        for i = 1, #exp.agent_target do
            a:add(exp.agent_target[i])
        end
        local params_target, gradParams_target = a:getParameters()

        log.infof('Creating target model(s), params=%d', params_target:nElement())

        return params, gradParams, params_target, gradParams_target
    end

    function exp.id(step_i, agent_i)
        return (step_i - 1) * opt.game_nagents + agent_i
    end

    function exp.stats(opt, game, stats, e)

        if e % opt.step_test == 0 then
            local test_idx = (e / opt.step_test - 1) % (opt.step / opt.step_test) + 1

            -- Initialise
            stats.test_opt = stats.test_opt or torch.zeros(opt.step / opt.step_test, opt.game_nagents)
            stats.test_god = stats.test_god or torch.zeros(opt.step / opt.step_test, opt.game_nagents)

            -- Naive strategy
            --local r_naive = 0
            --for b = 1, opt.bs do
            --    local has_been = game.has_been[{ { b }, { 1, opt.nsteps }, {} }]:sum(2):squeeze(2):gt(0):float():sum()
            --   if has_been == opt.game_nagents then
            --        r_naive = r_naive + game.reward_all_live
            --    else
            --        r_naive = r_naive + game.reward_all_die
            --    end
            --end
            --stats.test_opt[test_idx] = r_naive / opt.bs

            -- God strategy
            local r_god = 0
            for b = 1, opt.bs do
                local cPull = 2*game.correctPulls[b]
                r_god = r_god + game.reward_all_live * cPull
            end
            stats.test_god[test_idx] = r_god / opt.bs
        end


        -- Keep stats
        if e == opt.step then
            stats.test_opt_avg = stats.test_opt:mean()
            stats.test_god_avg = stats.test_god:mean()
        elseif e % opt.step == 0 then
            local coef = 0.9
            stats.test_opt_avg = stats.test_opt_avg * coef + stats.test_opt:mean() * (1 - coef)
            stats.test_god_avg = stats.test_god_avg * coef + stats.test_god:mean() * (1 - coef)
        end

        -- Print statistics
        if e % opt.step == 0 then
            log.infof('te_opt=%.2f, te_opt_avg=%.2f, te_god=%.2f, te_god_avg=%.2f',
                stats.test_opt:mean(),
                stats.test_opt_avg,
                stats.test_god:mean(),
                stats.test_god_avg)
        end
    end

    local function create_agent()



        -- Process inputs
        local model_input = nn.Sequential()
        model_input:add(nn.CAddTable(2))
        model_input:add(nn.BatchNormalization(opt.decision_model_size))

        local model_state = nn.Sequential()
        model_state:add(nn.LookupTable(opt.nsteps+1, opt.decision_model_size))

        -- Output
        local model_out = nn.Sequential()
        --if opt.upper_model_dropout > 0 then model_out:add(nn.Dropout(opt.model_dropout)) end
 --       model_out:add(nn.Linear(opt.decision_model_size, opt.decision_model_size))
        model_out:add(nn.ReLU(true))
        model_out:add(nn.Linear(opt.decision_model_size, opt.game_upper_action_space_total))

        -- Construct Graph
        local in_state = nn.Identity()()
        local in_id = nn.Identity()()


        local in_all = {
	    model_state(in_state),
            nn.LookupTable(opt.game_nagents, opt.decision_model_size)(in_id)
        }


        -- Process inputs
        local proc_input = model_input(in_all)

        -- Process out
        local proc_out = model_out(proc_input)

        -- Create model
        local model_inputs = { in_state, in_id}
        local model_outputs = { proc_out }


        nngraph.annotateNodes()

        local model = nn.gModule(model_inputs, model_outputs)

  --graph.dot(model.fg, 'MLP')

        return model:type(opt.dtype)
    end

    -- Create model
    local agent = create_agent()
    local agent_target = agent:clone()

    -- Knowledge sharing
    if opt.model_know_share == 1 then
        exp.agent = util.cloneManyTimes(agent, opt.game_nagents * 2)
        exp.agent_target = util.cloneManyTimes(agent_target, opt.game_nagents * 2)
    else
        exp.agent = {}
        exp.agent_target = {}

        local agent_copies = util.copyManyTimes(agent, opt.game_nagents)
        local agent_target_copies = util.copyManyTimes(agent_target, opt.game_nagents)

        for i = 1, opt.game_nagents do
            local unrolled = util.cloneManyTimes(agent_copies[i], 2)
            local unrolled_target = util.cloneManyTimes(agent_target_copies[i], 2)
            for s = 1, opt.nsteps + 1 do
                exp.agent[exp.id(s, i)] = unrolled[s]
                exp.agent_target[exp.id(s, i)] = unrolled_target[s]
            end
        end
    end

    return exp
end
