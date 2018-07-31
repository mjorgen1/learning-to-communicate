require 'torch'
local class = require 'class'

local log = require 'include.log'
local kwargs = require 'include.kwargs'
local util = require 'include.util'


local SimplePlan = class('SimplePlan')

-- Actions
-- 1 = stay
-- 2 = forward
-- 3 = backward
-- 4* = pull

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

    self.reward_option = 'optimisable' -- 'time-changing' 'easy'

    -- Spawn new game
    self:reset(1)
end

function SimplePlan:reset(episode)

    --save episode
    self.episode = episode

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
	    self.lever_pos[{ { b }, { agent } }] = torch.random(2,4) --lever position 1 means, there is no lever
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

        for b = 1, self.opt.bs do
            if self.agent_pos[b][agent] == self.lever_pos[b][agent] and self.lever_pos[b][agent] ~= 1 then
                range[b] = { { b }, { 1, bound } }
            elseif self.step_counter == 1 then
                range[b] = { { b }, { 1 , 1} }
	    else
                range[b] = { { b }, { 1 , bound -1} }
            end
        end
        return range
    else					--the rial option was not updated to fit the SimplePlan game yet
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


function SimplePlan:getCommLimited(step, i)
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

function SimplePlan:getReward(a_t,episode)
    
    self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

    for b = 1, self.opt.bs do
 
        if self.reward_option == 'easy' then

	    if self.terminal[b]==0 and (a_t[b][1] == 4 and a_t[b][2] == 4) then -- both did pull
                self.reward[b] = self.reward_all_live
		self.terminal[b] = 1
		--print('both pulled')
	    elseif self.terminal[b]==0 and (a_t[b][1] == 4 or a_t[b][2] == 4) then
		--self.reward[b] = self.reward_all_die
		self.terminal[b] = 1
		--print('one pulled')
	    end

	elseif self.reward_option == 'optimisable' then

	    if self.terminal[b]==0 and (a_t[b][1] == 4 and a_t[b][2] == 4) then -- both did pull
		earliest = torch.max(self.lever_pos[b])
                self.reward[b] = self.reward_all_live * 1/(self.step_counter-earliest)
		self.terminal[b] = 1
		--print('both pulled')
	    elseif self.terminal[b]==0 and (a_t[b][1] == 4 or a_t[b][2] == 4) then
		--self.reward[b] = self.reward_all_die
		self.terminal[b] = 1
		--print('one pulled')
	    end


        elseif self.reward_option == 'time-changing' then

    	    --reward for staying in communication distance at the beginning of the episode, reduces over steps and episodes
	    if self.terminal[b] == 0 and self.agent_pos[b]:sum(1)[1] == 2 and self.step_counter > 1 then
	        self.reward[b] = 1/(self.step_counter^2)/(1 + episode/200)
	        --if b == 1 then print('reward for communication distance') end
	    end

	    --rewards for pulling the lever, without coordination. Decreases into negative values over episodes

            if self.terminal[b] == 0 and self.pulled_lever[b]:sum(1)[1] == 0 then -- noone pulled by now

	        if (a_t[b][1] == 4 and a_t[b][2] == 4) then -- both did pull
                    self.reward[b] = self.reward_all_live
		    self.terminal[b] = 1
		    --if b == 1 then print('reward for both pulled') end
	        elseif (a_t[b][1] == 4 and a_t[b][2] ~= 4) then -- agent 1 did pull
		    self.reward[b] = self.reward_all_live * 1/(1+episode/2000) + self.reward_all_die * (1-1/(1+episode/2000))
		    self.pulled_lever[b][1] = 1
		    --if b == 1 then print('reward for agent 1 pulled') end
	        elseif (a_t[b][1] ~= 4 and a_t[b][2] == 4) then -- agent 2 did pull
		    self.reward[b] = self.reward_all_live * 1/(1+episode/2000) + self.reward_all_die * (1-1/(1+episode/2000))
		    self.pulled_lever[b][2] = 1
		    --if b == 1 then print('reward for agent 2 pulled') end
	        else
		    --if b == 1 then print('reward for none pulled') end
	        end

	    --reward for pulling the lever after the other, stops negative reward

            elseif self.terminal[b] == 0 and self.pulled_lever[b]:sum(1)[1] == 1 then -- one pulled by now

	        if (self.pulled_lever[b][1] == 1 and a_t[b][2] == 4) or (a_t[b][1] == 4 and self.pulled_lever[b][2] == 1) then -- both did pull
		    self.terminal[b] = 1
		    --if b == 1 then print('reward for second pulled') end
	        else
		    self.reward[b] = self.reward_small_off * 1/(1+episode/200)
		    --if b == 1 then print('reward for waiting for second pulled') end
	        end 
	    else
		if b == 1 then print('something went wrong or terminated'..self.terminal[b] == 1) end

	    end

	end

        if self.step_counter == self.opt.nsteps and self.terminal[b] == 0 then
            self.terminal[b] = 1
	    --print('0')
        end

    end
    return self.reward:clone(), self.terminal:clone()
end

function SimplePlan:step(a_t,episode)

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


function SimplePlan:getState()
    local state = {}

    for agent = 1, self.opt.game_nagents do
        state[agent] = torch.Tensor(self.opt.bs,2)

        for b = 1, self.opt.bs do
--	    if (self.agent_pos[b][agent] == 1) then
		state[agent][{{b}, {1}}]= self.lever_pos[b][agent]
--	    else
--		state[agent][{{b}, {1}}]= 1	
--	    end
	    state[agent][{{b},{2}}]= self.agent_pos[b][agent]
	    --print(self.agent_pos[b][agent])
        end
    end

    return state
end

function SimplePlan:imitateAction()
    local step = self.step_counter
    local pAction = torch.zeros(self.opt.bs, self.opt.game_nagents):type(self.opt.dtype)

    for b = 1, self.opt.bs do
        for agent = 1, self.opt.game_nagents do

            if (step == 1) then --if on first step, stay in place so comm occurs
                pAction[b][agent] = 1 
            elseif (step <= self.opt.nsteps) then
                if (self.agent_pos[b][agent] < self.lever_pos[b][agent]) then --if agent is behind lever, move forward
                    pAction[b][agent] = 2 
                elseif (self.agent_pos[b][agent] > self.lever_pos[b][agent]) then --if agent is ahead of lever, move backward
                    pAction[b][agent] = 3
                elseif (self.agent_pos[b][agent] == self.lever_pos[b][agent]) then --if agent at lever, functionality only for 2 agents
                    if (agent == 1) then
                        if (self.agent_pos[b][2] == self.lever_pos[b][2] and self.lever_pos[b][1] ~= 1 and self.lever_pos[b][2] ~= 1 ) then
                            pAction[b][agent] = 4 --pull if the other agent is at its lever too
                        else
                            pAction[b][agent] = 1 --stay put
                        end
                    elseif (agent == 2) then
                        if (self.agent_pos[b][1] == self.lever_pos[b][1] and self.lever_pos[b][1] ~= 1 and self.lever_pos[b][2] ~= 1) then
                            pAction[b][agent] = 4 --pull if the other agent is at its lever too
                        else
                            pAction[b][agent] = 1 --stay put
                        end
                    else
                        print("the number of agents should only be 2, check opt")
                    end
                end
            end

        end
    end

    return pAction
end

return SimplePlan

