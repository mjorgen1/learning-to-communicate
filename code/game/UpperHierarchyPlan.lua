require 'torch'
local class = require 'class'

local log = require 'include.log'
local kwargs = require 'include.kwargs'
local util = require 'include.util'


local UpperHierarchyPlan = class('UpperHierarchyPlan')

-- Actions
-- 1 = wait
-- 2 = pull


function UpperHierarchyPlan:__init(opt)

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
    self.reward_all_die = -3 + self.opt.game_reward_shift
    self.reward_small_off = -0.0

    self.reward_option =  'optimisable' -- 'time-changing'  'easy'

    -- Spawn new game
    self:reset(1)
end

function UpperHierarchyPlan:reset(episode)

    --save episode
    self.episode = episode

    -- Reset rewards
    self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

    self.upper_reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

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

    -- Whos UpperHierarchyPlan is at which position? 
    self.lever_pos = torch.zeros(self.opt.bs, self.opt.game_nagents)

    lever_pos_distribution = {2,2,3,4}

    for b = 1, self.opt.bs do
	for agent = 1, self.opt.game_nagents do
	    index = torch.random(1,4) 
	    self.lever_pos[{ { b }, { agent } }] = lever_pos_distribution[index]
        end

    end


    return self
end

function UpperHierarchyPlan:getLowerActionRange(step, agent, time_target)
    local range = {}
    if self.opt.model_dial == 1 then           
        local bound = self.opt.game_lower_action_space

        for b = 1, self.opt.bs do
            if self.step_counter == time_target[b][agent] then
                range[b] = { { b }, { 1, bound } }
	    else
                range[b] = { { b }, { 1 , 1} }
            end
        end
        return range
    else					--the rial option was not updated to fit the UpperHierarchyPlan game yet
        local comm_range = {}
        for b = 1, self.opt.bs do
            if self.pulled_lever[b][agent] == 0 then
                range[b] = { { b }, { 1, self.opt.game_lower_action_space } }              
            else
                range[b] = { { b }, { 1 } }
            end
	    comm_range[b] = { { b }, { self.opt.game_lower_action_space + 1, self.opt.game_lower_action_space_total } }
        end
        return range, comm_range
    end
end


function UpperHierarchyPlan:getCommLimited(step, i)
    if self.opt.game_comm_limited == 1 then

        local range = {}

        -- Get range per batch
        for b = 1, self.opt.bs do
            if step >= 2 then
                if i == 1  then
                    range[b] = { 2, {} }
                elseif  i == 2  then
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

function UpperHierarchyPlan:getReward(a_t,episode)
    
    self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

    for b = 1, self.opt.bs do
 	if self.terminal[b]==0 then 
	    earliest = torch.max(self.lever_pos[b])
	    if a_t[b][1] ~= 1 then
		self.pulled_lever[b][1] = 1
	        if a_t[b][1] == earliest then
                    self.reward[b] = self.reward[b] + self.reward_all_live
		else
		    self.reward[b] = self.reward[b] + self.reward_all_die
		end
	    else
		self.reward[b] = self.reward[b] + self.reward_small_off
	    end
	    if a_t[b][2] ~= 1 then
		self.pulled_lever[b][2] = 1
	        if a_t[b][2] == earliest then
                    self.reward[b] = self.reward[b] + self.reward_all_live
		else
		    self.reward[b] = self.reward[b] + self.reward_all_die
		end
	    else
		self.reward[b] = self.reward[b] + self.reward_small_off
	    end

	    if (self.pulled_lever[b][1] == 1 and self.pulled_lever[b][2] == 1) then
		self.terminal[b] = 1
	    end

	end

	self.upper_reward[b] = self.upper_reward[b] + self.reward[b]
    end

    return self.reward:clone(), self.terminal:clone()
end

function UpperHierarchyPlan:lower_step(a_t,episode)

    -- Get rewards
    local reward, terminal = self:getReward(a_t,episode)

    -- Make step
    self.step_counter = self.step_counter + 1

    return reward, terminal
end

function UpperHierarchyPlan:upper_step(a_t,episode)
    return self.upper_reward:clone()
end



function UpperHierarchyPlan:getState()
    local state = {}

    for agent = 1, self.opt.game_nagents do
        state[agent] = torch.Tensor(self.opt.bs,2)

        for b = 1, self.opt.bs do
	    state[agent][{{b}, {1}}]= self.lever_pos[b][agent]
	    state[agent][{{b},{2}}]= self.step_counter
        end
    end

    return state
end

function UpperHierarchyPlan:imitateAction()
    local step = self.step_counter
    local pAction = torch.zeros(self.opt.bs, self.opt.game_nagents):type(self.opt.dtype)

    if self.opt.model_dial == 1 then

        for b = 1, self.opt.bs do
            for agent = 1, self.opt.game_nagents do

                if (step == 3) then --if on first step, stay in place so comm occurs : TODO action_space/(comm_bits^2)
		    earliest = torch.max(self.lever_pos[b])
		    pAction[b][agent] = earliest 
                
                else 
		    pAction[b][agent] = 1 
               
                end

            end
        end

        return pAction

    else
	local commAction = torch.zeros(self.opt.bs, self.opt.game_nagents):type(self.opt.dtype)

        for b = 1, self.opt.bs do
            for agent = 1, self.opt.game_nagents do
	
		if step == 1 then
		    if self.lever_pos[b][agent] > 3 then
		        pAction[b][agent] = 4
                        commAction[b][agent] = 6
		    else
		    	pAction[b][agent] = 1
                    	commAction[b][agent] = 5
		    end

                elseif (step == 2 and self.terminal[b] == 0) then 

		    if torch.max(self.lever_pos[b]) == 4 and self.pulled_lever[b][agent] == 0 then
		    	pAction[b][agent] = 4
                    	commAction[b][agent] = 5
		    
		    elseif self.lever_pos[b][agent] > 2 then
		        pAction[b][agent] = 3
                        commAction[b][agent] = 6
		    else
		    	pAction[b][agent] = 1
                    	commAction[b][agent] = 5
		    end
                
                elseif (step == 3 and self.terminal[b] == 0) then 

		    if torch.max(self.lever_pos[b]) == 3 and self.pulled_lever[b][agent]== 0 then
		    	pAction[b][agent] = 3
                    	commAction[b][agent] = 5
		    
		    elseif self.lever_pos[b][agent] == 2 then
		        pAction[b][agent] = 2
                        commAction[b][agent] = 5
		    else
		    	pAction[b][agent] = 1
                    	commAction[b][agent] = 5
		    end
		else
		    pAction[b][agent] = 1
                    commAction[b][agent] = 5
                end

            end
        end

        return commAction, pAction
    end
end

return UpperHierarchyPlan

