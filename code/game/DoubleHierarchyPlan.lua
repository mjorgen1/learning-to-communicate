require 'torch'
local class = require 'class'

local log = require 'include.log'
local kwargs = require 'include.kwargs'
local util = require 'include.util'


local HierarchyPlan = class('HierarchyPlan')

-- Meta Actions
-- 1 = 0 steps of comm
-- 2 = 1 step of comm
-- 3 = 2 steps of comm
-- ...


-- Actions
-- 1 = stay
-- 2 = forward
-- 3* = pull

function HierarchyPlan:__init(opt)

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
    self:reset(1)
end

function HierarchyPlan:reset(episode)

    --save episode
    self.episode = episode

    -- Reset rewards
    self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

    self.upper_reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

    self.time_target = torch.zeros(self.opt.bs, self.opt.game_nagents)
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

    -- Whos HierarchyPlan is at which position? 
    self.lever_pos = torch.zeros(self.opt.bs, self.opt.game_nagents)

    lever_pos_distribution = {2,2,2,2,2,2,3,3,3,4}

    for b = 1, self.opt.bs do
	for agent = 1, self.opt.game_nagents do
	    index = torch.random(1,10) 
	    self.lever_pos[{ { b }, { agent } }] = lever_pos_distribution[index]
        end

    end

    self.pot_weight = self.opt.gamma

    return self
end

function HierarchyPlan:getUpperActionRange(step, agent, max)
    local range = {}
    if self.opt.model_dial == 1 then           
        local bound = self.opt.game_upper_action_space

        for b = 1, self.opt.bs do
	    if step == max[b][agent] then
                range[b] = { { b }, { 1, bound } }
	    else
                range[b] = { { b }, { 1 , 1} }
            end
        end
        return range
    else					--the rial option was not updated to fit the HierarchyPlan game yet
        local comm_range = {}
        for b = 1, self.opt.bs do
            if self.active_agent[b][step] == agent then
                range[b] = { { b }, { 1, self.opt.game_action_space } }
                comm_range[b] = { { b }, { self.opt.game_action_space + 1, self.opt.game_action_space_total } }
            else
                range[b] = { { b }, { 1 } }
                comm_range[b] = { { b }, { 0, 0 } }
            end
        end
        return range, comm_range
    end
end

function HierarchyPlan:getMetaActionRange(step, agent)
    local range = {}
    if self.opt.model_dial == 1 then           
        local bound = self.opt.game_meta_action_space

        for b = 1, self.opt.bs do
            if step == 1 then
                range[b] = { { b }, { 1, bound } }
	    else
                range[b] = { { b }, { 1 , 1} }
            end
        end
        return range
    else					--the rial option was not updated to fit the HierarchyPlan game yet
        local comm_range = {}
        for b = 1, self.opt.bs do
            if self.active_agent[b][step] == agent then
                range[b] = { { b }, { 1, self.opt.game_action_space } }
                comm_range[b] = { { b }, { self.opt.game_action_space + 1, self.opt.game_action_space_total } }
            else
                range[b] = { { b }, { 1 } }
                comm_range[b] = { { b }, { 0, 0 } }
            end
        end
        return range, comm_range
    end
end

function HierarchyPlan:getLowerActionRange(step, agent)
    local range = {}
    if self.opt.model_dial == 1 then           
        local bound = self.opt.game_lower_action_space

        for b = 1, self.opt.bs do
            if self.agent_pos[b][agent] == self.lever_pos[b][agent] and self.lever_pos[b][agent] ~= 1 then
                range[b] = { { b }, { 1, bound } }
            --elseif self.step_counter == 1 then
                --range[b] = { { b }, { 1 , 1} }
	    else
                range[b] = { { b }, { 1 , bound -1} }
            end
        end
        return range
    else					--the rial option was not updated to fit the HierarchyPlan game yet
        local comm_range = {}
        for b = 1, self.opt.bs do
            if self.active_agent[b][step] == agent then
                range[b] = { { b }, { 1, self.opt.game_action_space } }
                comm_range[b] = { { b }, { self.opt.game_action_space + 1, self.opt.game_action_space_total } }
            else
                range[b] = { { b }, { 1 } }
                comm_range[b] = { { b }, { 0, 0 } }
            end
        end
        return range, comm_range
    end
end

function HierarchyPlan:getReward(a_t,time_target)
    
    self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

    for b = 1, self.opt.bs do

	if self.terminal[b] == 0 and self.pulled_lever[b]:sum(1)[1] == 0 then -- noone pulled by now

	    if self.terminal[b]==0 and a_t[b][1] == 3 and a_t[b][2] ~= 3 then
		if self.step_counter == time_target[b][1]+1 then
		    self.reward[b][1] = self.reward_all_live
		elseif self.step_counter < time_target[b][1] or self.step_counter > time_target[b][1]+2 then
		    self.reward[b][1] = -self.reward_all_live
		end
		self.pulled_lever[b][1] = 1

            elseif self.terminal[b]==0 and a_t[b][1] ~= 3 and a_t[b][2] == 3 then
		if self.step_counter == time_target[b][2]+1 then
		    self.reward[b][2] = self.reward_all_live
		elseif self.step_counter < time_target[b][2] or self.step_counter > time_target[b][2]+2 then
		    self.reward[b][2] = -self.reward_all_live
		end
		self.pulled_lever[b][2] = 1


	    --cooperative reward for successfully pulling the lever
	    elseif self.terminal[b]==0 and (a_t[b][1] == 3 and a_t[b][2] == 3) then
	        earliest = torch.max(self.lever_pos[b])
	        if (self.step_counter == earliest) then
               	    self.upper_reward[b] = self.reward_all_live
	        end

		if self.step_counter == time_target[b][1]+1 then
		    self.reward[b][1] = self.reward_all_live
		elseif self.step_counter < time_target[b][1] or self.step_counter > time_target[b][1]+2 then
		    self.reward[b][1] = - self.reward_all_live
		end

		if self.step_counter == time_target[b][2]+1 then
		    self.reward[b][2] = self.reward_all_live
		elseif self.step_counter < time_target[b][2] or self.step_counter > time_target[b][2]+2 then
		    self.reward[b][2] = - self.reward_all_live
		end
	        self.terminal[b] = 1
	    end


        elseif self.terminal[b] == 0 and self.pulled_lever[b]:sum(1)[1] == 1 then -- one pulled by now
	    if (self.pulled_lever[b][1] == 1 and a_t[b][2] == 3) then
		if self.step_counter == time_target[b][2]+1 then
		    self.reward[b][2] = self.reward_all_live
		elseif self.step_counter < time_target[b][2] or self.step_counter > time_target[b][2]+2 then
		    self.reward[b][2] = -self.reward_all_live
		end
		self.terminal[b] = 1
	    elseif (a_t[b][1] == 3 and self.pulled_lever[b][2] == 1) then -- both did pull
		if self.step_counter == time_target[b][1]+1 then
		    self.reward[b][1] = self.reward_all_live
		elseif self.step_counter < time_target[b][1] or self.step_counter > time_target[b][1]+2 then
		    self.reward[b][1] = -self.reward_all_live
		end
		self.terminal[b] = 1
	    end
	end

        if self.step_counter == self.opt.nsteps and self.terminal[b] == 0 then
            self.terminal[b] = 1
        end

    end

    return self.reward:clone(), self.terminal:clone()
end

function HierarchyPlan:lower_step(a_t,time_target)

    self.time_target = time_target

    -- Get rewards
    local reward, terminal = self:getReward(a_t,time_target)


    -- Make step
    self.step_counter = self.step_counter + 1

    for b = 1, self.opt.bs do
        for agent = 1, self.opt.game_nagents do

	    local movement = 0

	    if a_t[b][agent] == 2 then
		movement = 1
		--print(agent .. ' moved forward')
	    end

            self.agent_pos[b][agent] = self.agent_pos[b][agent] + movement
        end
    end

    return reward, terminal
end

function HierarchyPlan:upper_step()
    return self.upper_reward:clone()
end


function HierarchyPlan:getState()
    local state = {}

    for agent = 1, self.opt.game_nagents do
        state[agent] = torch.Tensor(self.opt.bs,2)

        for b = 1, self.opt.bs do
	    state[agent][{{b},{1}}]= self.lever_pos[b][agent]
	    state[agent][{{b},{2}}]= self.agent_pos[b][agent]

        end
    end

    return state
end

function HierarchyPlan:getEmptyState()
    local state = {}

    for agent = 1, self.opt.game_nagents do
        state[agent] = torch.Tensor(self.opt.bs,2)

        for b = 1, self.opt.bs do
	    state[agent][{{b},{1}}]= 1
	    state[agent][{{b},{2}}]= 1

        end
    end

    return state
end

function HierarchyPlan:imitateAction()
    local step = self.step_counter
    local pAction = torch.zeros(self.opt.bs, self.opt.game_nagents):type(self.opt.dtype)
    
    for b = 1, self.opt.bs do
        for agent = 1, self.opt.game_nagents do
            if (step <= self.opt.nsteps) then
                if (self.agent_pos[b][agent] < self.lever_pos[b][agent]) then --if agent is behind lever, move forward
                    pAction[b][agent] = 2 
                elseif (self.agent_pos[b][agent] == self.lever_pos[b][agent]) then --if agent at lever
                    if (step == self.time_target[b][agent]+1) then
                        pAction[b][agent] = 3 --pull if at the correct time
                    else
                        pAction[b][agent] = 1 --stay put, until its the time to pull
                    end
                end
            end
        end
    end

    return pAction
end

return HierarchyPlan

