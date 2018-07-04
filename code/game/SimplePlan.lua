require 'torch'
local class = require 'class'

local log = require 'include.log'
local kwargs = require 'include.kwargs'
local util = require 'include.util'


local SimplePlan = class('SimplePlan')

-- Actions
-- 1 = on
-- 2 = off
-- 3 = tell
-- 4* = none

function SimplePlan:__init(opt)
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
    self.reward_small_off = -0.2

    -- Spawn new game
    self:reset()
end

function SimplePlan:reset()

    -- Reset rewards
    self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

    --Reset correctPulls
    self.correctPulls = torch.zeros(self.opt.bs)

    -- Reached end
    self.terminal = torch.zeros(self.opt.bs)

    -- Step counter
    self.step_counter = 1

    -- Was one lever pulled
    self.pulled_lever = torch.zeros(self.opt.bs, self.opt.game_nagents)

    -- Who is in
    self.active_agent = torch.zeros(self.opt.bs, self.opt.nsteps, self.opt.game_nagents)

    -- Agent positions
    self.agent_pos = torch.ones(self.opt.bs,self.opt.game_nagents)

    -- Whos SimplePlan is at which position? 
    self.lever_pos = torch.zeros(self.opt.bs, self.opt.game_nagents)
    for b = 1, self.opt.bs do
	for agent = 1, self.opt.game_nagents do
            self.lever_pos[{ { b }, { agent } }] = torch.random(2,self.opt.nsteps-2)
        end

    end

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

function SimplePlan:getActionRange(step, agent)
    local range = {}
    if self.opt.model_dial == 1 then           
        local bound = self.opt.game_action_space

        for i = 1, self.opt.bs do
            if self.agent_pos[i][agent] == self.lever_pos[i][agent] then
                range[i] = { { i }, { 1, bound } }
            else
                range[i] = { { i }, { 1 , bound -1} }
            end
        end
        return range
    else					--the rial option was not updated to fit the SimplePlan game yet
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


function SimplePlan:getCommLimited(step, i)
    if self.opt.game_comm_limited then

        local range = {}

        -- Get range per batch
        for b = 1, self.opt.bs do
            if self.agent_pos[b]:sum(1)[1] == 0 then
                if step > 1 and i == 1 then
                    range[b] = { 2, {} }
                elseif step > 1 and i == 2 then
                    range[b] = { 1, {} }
	        else
                    range[b] = 0
                end
	    else
                range[b] = 0
	    end
        end
        return range
    else
        return nil
    end
end

function SimplePlan:getReward(a_t)
    for b = 1, self.opt.bs do

        if self.terminal[b] == 0 and self.pulled_lever[b]:sum(1)[1] == 0 then -- noone pulled by now

	    if (a_t[b][1] == 4 and a_t[b][2] == 4) then -- both did pull
                self.reward[b] = self.reward_all_live
		self.terminal[b] = 1
	    elseif (a_t[b][1] == 4 and a_t[b][2] ~= 4) then -- agent 1 did pull
		self.reward[b] = self.reward_all_live
		self.pulled_lever[b][1] = 1
	    elseif (a_t[b][1] ~= 4 and a_t[b][2] == 4) then -- agent 2 did pull
		self.reward[b] = self.reward_all_live
		self.pulled_lever[b][2] = 1
	    end

        elseif self.terminal[b] == 0 and self.pulled_lever[b]:sum(1)[1] == 1 then -- one pulled by now

	    if (self.pulled_lever[b][1] == 1 and a_t[b][2] == 4) or (a_t[b][1] == 4 and self.pulled_lever[b][2] == 1) then -- both did pull
		self.terminal[b] = 1
	    else
		self.reward[b] = self.reward_small_off
	    end

	end
        if self.step_counter == self.opt.nsteps and self.terminal[b] == 0 then
            self.terminal[b] = 1
        end

    end
    return self.reward:clone(), self.terminal:clone()
end

function SimplePlan:step(a_t)

    -- Get rewards
    local reward, terminal = self:getReward(a_t)


    -- Make step
    self.step_counter = self.step_counter + 1

    for b = 1, self.opt.bs do
        for agent = 1, self.opt.game_nagents do

	    local movement = 0

	    if a_t[b][agent] == 2 then
		movement = 1
		--print(agent .. ' moved forward')
	    elseif a_t[b][agent] == 3 and self.agent_pos[b][agent] > 1 then
		movement = -1
		--print(agent .. ' moved backward')
	    end

            self.agent_pos[b][agent] = self.agent_pos[b][agent] + movement
        end
    end

    return reward, terminal
end


function SimplePlan:getState()
    local state = {}

    for agent = 1, self.opt.game_nagents do
        state[agent] = torch.Tensor(self.opt.bs)

        for b = 1, self.opt.bs do
	    if (self.step_counter == 1) then
		state[agent][{{b}}]= self.lever_pos[b][agent]
	    else
		state[agent][{{b}}]= self.agent_pos[b][agent]
	    end
	    --print(self.agent_pos[b][agent])
        end
    end

    return state
end

return SimplePlan

