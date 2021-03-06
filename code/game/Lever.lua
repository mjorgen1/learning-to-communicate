require 'torch'
local class = require 'class'

local log = require 'include.log'
local kwargs = require 'include.kwargs'
local util = require 'include.util'


local Lever = class('Lever')

-- Actions
-- 1 = on
-- 2 = off
-- 3 = tell
-- 4* = none

function Lever:__init(opt)
    local opt_game = kwargs(_, {
        { 'game_action_space', type = 'int-pos', default = 2 },
        { 'game_reward_shift', type = 'int', default = 0 },
        { 'game_comm_bits', type = 'int', default = 0 },
        { 'game_comm_sigma', type = 'number', default = 2 },
    })



    for k, v in pairs(opt_game) do
        if not opt[k] then
            opt[k] = v
        end
    end
    self.opt = opt

    -- Rewards
    self.reward_all_live = 1 + self.opt.game_reward_shift
    self.reward_all_die = -1 + self.opt.game_reward_shift

    -- Spawn new game
    self:reset()
end

function Lever:reset()

    -- Reset rewards
    self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

    --Reset correctPulls
    self.correctPulls = torch.zeros(self.opt.bs)

    -- Reached end
    self.terminal = torch.zeros(self.opt.bs)

    -- Step counter
    self.step_counter = 1

    -- Who is in
    self.active_agent = torch.zeros(self.opt.bs, self.opt.nsteps, self.opt.game_nagents)

    for b = 1, self.opt.bs do
        for step = 1, self.opt.nsteps do
	        for agent = 1, self.opt.game_nagents do
                self.active_agent[{ { b }, { step } , { agent } }] = torch.random(1, 2)
            end
            for i = 1, self.opt.game_nagents do
                if(self.active_agent[b][step][i] == 2) then
                    self.correctPulls[b] = self.correctPulls[b] + 1
                end
            end
        end
    end

    return self
end

function Lever:getActionRange(step, agent)
    local range = {}
    if self.opt.model_dial == 1 then           
        local bound = self.opt.game_action_space

        for i = 1, self.opt.bs do
            if self.active_agent[i][step][agent] == 2 and step > 1 then
                range[i] = { { i }, { 1, bound } }
            else
                range[i] = { { i }, { 1 } }
            end
        end
        return range
    else					--the rial option was not updated to fit the Lever game yet
        local comm_range = {}
        for i = 1, self.opt.bs do
            if self.active_agent[i][step] == agent then
                range[i] = { { i }, { 1, self.opt.game_action_space } }
                comm_range[i] = { { i }, { self.opt.game_action_space + 1, self.opt.game_action_space_total } }
            else
                range[i] = { { i }, { 1 } }
                comm_range[i] = { { i }, { 0, 0 } }
            end
        end
        return range, comm_range
    end
end


function Lever:getCommLimited(step, i)
    if self.opt.game_comm_limited then

        local range = {}

        -- Get range per batch
        for b = 1, self.opt.bs do
            if step > 1 then
                range[b] = {{1, self.opt.game_nagents}, {}}
            else
                range[b] = 0
            end
        end
        return range
    else
        return nil
    end
end

function Lever:getReward(a_t)
    for b = 1, self.opt.bs do
        won = false 
        for i = 1, self.opt.game_nagents do
            if(a_t[b][1] == 2) then
                won = true
            else
                won = false
                break
            end
        end
        if won then
            self.reward[b] = self.reward_all_live
        else
            self.reward[b] = self.reward_all_die
	    end 
        if self.step_counter == self.opt.nsteps and self.terminal[b] == 0 then
            self.terminal[b] = 1
        end
    end
    return self.reward:clone(), self.terminal:clone()
end

function Lever:step(a_t)

    -- Get rewards
    local reward, terminal = self:getReward(a_t)


    -- Make step
    self.step_counter = self.step_counter + 1

    return reward, terminal
end


function Lever:getState()
    local state = {}

    for agent = 1, self.opt.game_nagents do
        state[agent] = torch.Tensor(self.opt.bs)

        for b = 1, self.opt.bs do
	    if (self.step_counter == self.opt.nsteps) then
		    state[agent][{ { b } }] = 2
	    else
            state[agent][{ { b } }] = self.active_agent[b][self.step_counter+1][agent]
	    end
        end
    end

    return state
end

return Lever

