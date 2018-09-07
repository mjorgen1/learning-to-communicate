require 'torch'
local class = require 'class'

local log = require 'include.log'
local kwargs = require 'include.kwargs'
local util = require 'include.util'


local SingleLeverPlan = class('SingleLeverPlan')

-- Actions
-- 1 = on
-- 2 = off
-- 3 = tell
-- 4* = none

function SingleLeverPlan:__init(opt)
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

    self.reward_option = 'easy' -- 'time-changing' 'optimisable'

    -- Spawn new game
    self:reset()
end

function SingleLeverPlan:reset()

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

    -- Whos Lever is at which position? 
    self.lever_pos = torch.zeros(self.opt.bs, self.opt.game_nagents)
    for b = 1, self.opt.bs do
	for agent = 1, self.opt.game_nagents do
            self.lever_pos[{ { b }, { agent } }] = torch.random(2,6)
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

function SingleLeverPlan:getActionRange(step, agent)
    local range = {}
    local comm_range = {}
       
        local bound = self.opt.game_action_space

        for b = 1, self.opt.bs do
            if self.agent_pos[b][agent] == self.lever_pos[b][agent] then
                range[b] = { { b }, { 1, bound } }
            else
                range[b] = { { b }, { 1 , bound -1} }
            end
            comm_range[b] = { { b }, { 0, 0 } }
	end

        return range, comm_range
end


function SingleLeverPlan:getCommLimited(step, i)
    if self.opt.game_comm_limited == 1 then

        local range = {}

        -- Get range per batch
        for b = 1, self.opt.bs do
            if self.agent_pos[b]:sum(1)[1] == 2 then
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
        local range = {}

        -- Get range per batch
        for b = 1, self.opt.bs do
            if step > 1 and i == 1 then
                range[b] = { 2, {} }
            elseif step > 1 and i == 2 then
                range[b] = { 1, {} }
	    else
                range[b] = 0
            end
        end
        return range
    end
end

function SingleLeverPlan:getReward(a_t,episode)
    
    self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

    for b = 1, self.opt.bs do

	if self.terminal[b]==0 and a_t[b][1] == 4 then -- did pull
            self.reward[b] = self.reward_all_live * 1/(self.step_counter - self.lever_pos[b][1] +1)
	    self.terminal[b] = 1
	    --print(self.reward[b][1] .. ' '.. self.lever_pos[b][1])
	end

        if self.step_counter == self.opt.nsteps and self.terminal[b] == 0 then
            self.terminal[b] = 1
	    --print(0 .. ' '.. self.lever_pos[b][1])
        end

    end
    return self.reward:clone(), self.terminal:clone()
end

function SingleLeverPlan:step(a_t,episode)

    -- Get rewards
    local reward, terminal = self:getReward(a_t,episode)


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


function SingleLeverPlan:getState()
    local state = {}

    for agent = 1, self.opt.game_nagents do
        state[agent] = torch.Tensor(self.opt.bs,2)

        for b = 1, self.opt.bs do
	    if (self.agent_pos[b][agent] == 1) then
		state[agent][{{b}, {1}}]= self.lever_pos[b][agent]
	    else
		state[agent][{{b}, {1}}]= 1	
	    end
	    state[agent][{{b},{2}}]= self.agent_pos[b][agent]
	    --print(agent ..' position: ' .. self.agent_pos[b][agent])
        end
    end

    return state
end

return SingleLeverPlan

