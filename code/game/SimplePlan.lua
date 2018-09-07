require 'torch'
local class = require 'class'

local log = require 'include.log'
local kwargs = require 'include.kwargs'
local util = require 'include.util'


local SimplePlan = class('SimplePlan')

-- Actions
-- 1 = stay
-- 2 = forward
-- 3* = pull

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

    self.reward_option =  'easy'--'potential'--'optimisable' -- 'time-changing' 

    -- Spawn new game
    self:reset(1)
end

function SimplePlan:reset(gradient_check)

    gradient_check = gradient_check or 0
    --save episode
    --self.episode = episode

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

    lever_pos_distribution = {2,2,3,4}

    for b = 1, self.opt.bs do
	for agent = 1, self.opt.game_nagents do
            index = torch.random(1,4) 
            if gradient_check ~= 1 or (b == 1) then
                self.lever_pos[{ { b }, { agent } }] = lever_pos_distribution[index]
            elseif (b > 1) and gradient_check == 1 then
                self.lever_pos[{ { b }, { agent } }] = self.lever_pos[{ { 1 }, { agent } }] 
            end
        end
    
    end
    self.pot_weight = self.opt.gamma

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
    else					--the rial option
        local comm_range = {}
        local bound = self.opt.game_action_space
        for b = 1, self.opt.bs do
            if self.agent_pos[b][agent] == self.lever_pos[b][agent] and self.lever_pos[b][agent] ~= 1 then
                range[b] = { { b }, { 1, bound } }
            --elseif self.step_counter == 1 then
            --    range[b] = { { b }, { 1 , 1} }
            else
                range[b] = { { b }, { bound - 1 } } 
            end
            if self.agent_pos[b]:sum(1)[1] == 2 then
                comm_range[b] = { { b }, { bound + 1, self.opt.game_action_space_total } }
            else
                comm_range[b] = { { b }, { 0, 0 } }
            end
        end
        return range, comm_range
    end
end


function SimplePlan:getCommLimited(step, i)
    if self.opt.game_comm_limited == 1 then --commLimited

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
    elseif 
	self.opt.game_comm_limited == 0 then --commLimited

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
    else                                --no commLimited
        return nil
    end
    --3 comm actions for a_t 
end


function SimplePlan:getEarliest()
    local earliest = torch.max(self.lever_pos[1])+1
    return earliest
end

function SimplePlan:getReward(a_t,episode)
    
    self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

    for b = 1, self.opt.bs do
 
        if self.reward_option == 'easy' then
	    local earliest = torch.max(self.lever_pos[b])+1
	    if self.terminal[b]==0 and (a_t[b][1] == 3 and a_t[b][2] == 3) then -- both did pull
                self.reward[b] = self.reward_all_live
		self.terminal[b] = 1
	    elseif self.terminal[b]==0 and (a_t[b][1] == 3 or a_t[b][2] == 3) then
	        --self.reward[b] = self.reward_all_die
	        self.terminal[b] = 1
            end

    	    if self.step_counter == earliest then
                self.terminal[b] = 1
            end

	elseif self.reward_option == 'optimisable' then

	    if self.terminal[b]==0 and (a_t[b][1] == 3 and a_t[b][2] == 3) then -- both did pull
	        earliest = torch.max(self.lever_pos[b])
	        if (self.step_counter > earliest) then  --both pulled but not at the first instance they could have
                    self.reward[b] = self.reward_all_live * 1/(self.step_counter-earliest)
		end
		self.terminal[b] = 1
		        --print('both pulled')
	    elseif self.terminal[b]==0 and (a_t[b][1] == 3 or a_t[b][2] == 3) then
		        --self.reward[b] = self.reward_all_die
		self.terminal[b] = 1
		        --print('one pulled')
	    end

        elseif self.reward_option == 'time-changing' then
    	    --reward for staying in communication distance at the beginning of the episode, reduces over steps and episodes
	        --[[ if self.terminal[b] == 0 and self.agent_pos[b]:sum(1)[1] == 2 and self.step_counter > 1 then
	            self.reward[b] = 1/(self.step_counter^2)/(1 + episode/200)
	        --if b == 1 then print('reward for communication distance') end
	        end--]]

	        --rewards for pulling the lever, without coordination. Decreases into negative values over episodes

            if self.terminal[b] == 0 and self.pulled_lever[b]:sum(1)[1] == 0 then -- noone pulled by now

                earliest = torch.max(self.lever_pos[b]) - 1
	            if (a_t[b][1] == 3 and a_t[b][2] == 3) then -- both did pull
                    self.reward[b] = self.reward_all_live * 1/(self.step_counter-earliest)
		            self.terminal[b] = 1
		            --if b == 1 then print('reward for both pulled') end
	            elseif (a_t[b][1] == 3 and a_t[b][2] ~= 3) then -- agent 1 did pull
		            self.reward[b] = self.reward_all_live* 1/(1+episode/2000) + self.reward_all_die * (1-1/(1+episode/2000))
		            self.pulled_lever[b][1] = 1
		            --if b == 1 then print('reward for agent 1 pulled') end
	            elseif (a_t[b][1] ~= 3 and a_t[b][2] == 3) then -- agent 2 did pull
		            self.reward[b] = self.reward_all_live * 1/(1+episode/2000) + self.reward_all_die * (1-1/(1+episode/2000))
		            self.pulled_lever[b][2] = 1
		            --if b == 1 then print('reward for agent 2 pulled') end
	            else
		            --if b == 1 then print('reward for none pulled') end
	            end

	        --reward for pulling the lever after the other, stops negative reward

            elseif self.terminal[b] == 0 and self.pulled_lever[b]:sum(1)[1] == 1 then -- one pulled by now

	            if (self.pulled_lever[b][1] == 1 and a_t[b][2] == 3) or (a_t[b][1] == 3 and self.pulled_lever[b][2] == 1) then -- both did pull
		            self.terminal[b] = 1
		            --if b == 1 then print('reward for second pulled') end
	            else
		            self.reward[b] = self.reward_small_off * 1/(1+episode/200)
		            --if b == 1 then print('reward for waiting for second pulled') end
	            end 

	        end

	    elseif self.reward_option == 'potential' then

	        --reward for the single agent giving a hint on the right path
	        if self.terminal[b]==0 and self.agent_pos[b][1] == 1 and self.agent_pos[b][2] == 1 then
		        --if a_t[b][1] == 2 then
		          --  self.reward[b][1] = self.reward[b][1] -1 * self.pot_weight
			--end
			--if a_t[b][2] == 2 then
		          --  self.reward[b][2] = self.reward[b][2] -1 * self.pot_weight
		        --end
	        elseif self.terminal[b]==0 then

		        if self.agent_pos[b][1] < self.lever_pos[b][1] then
		            if a_t[b][1]==2 then
			            self.reward[b][1] = self.reward[b][1] + 1 * self.pot_weight
		            end
		        elseif self.agent_pos[b][1] > self.lever_pos[b][1] then
		            if a_t[b][1]==2 then
			            self.reward[b][1] = self.reward[b][1] - 1 * self.pot_weight
		            end
		        elseif self.agent_pos[b][1] == self.lever_pos[b][1] then
		            if a_t[b][1]==2 then
			            self.reward[b][1] = self.reward[b][1] - 1 * self.pot_weight
		            elseif a_t[b][1]==1 then
			            if self.agent_pos[b][2] < self.lever_pos[b][2] then
			                self.reward[b][1] = self.reward[b][1] + 1 * self.pot_weight 
			            else
			                self.reward[b][1] = self.reward[b][1] -1 * self.pot_weight
			            end
		            elseif a_t[b][1]==3 then
			            if a_t[b][2]==3 then
			                self.reward[b][1] = self.reward[b][1] + 1 * self.pot_weight
			            elseif self.agent_pos[b][2]<self.lever_pos[b][2] then
			                self.reward[b][1] = self.reward[b][1] - 3 * self.pot_weight
			            elseif self.agent_pos[b][2]>=self.lever_pos[b][2] then
			                self.reward[b][1] = self.reward[b][1] + 0.5 * self.pot_weight
			            end
		            end
		        end

		        if self.agent_pos[b][2] < self.lever_pos[b][2] then
		            if a_t[b][2]==2 then
			            self.reward[b][2] = self.reward[b][2] + 1 * self.pot_weight
		            end
		        elseif self.agent_pos[b][2] > self.lever_pos[b][2] then
		            if a_t[b][2]==2 then
			            self.reward[b][2] = self.reward[b][2] - 1 * self.pot_weight
		            end
		        elseif self.agent_pos[b][2] == self.lever_pos[b][2] then
		            if a_t[b][2]==2 then
			            self.reward[b][2] = self.reward[b][2] - 1 * self.pot_weight
		            elseif a_t[b][2]==1 then
			            if self.agent_pos[b][1] < self.lever_pos[b][1] then
			                self.reward[b][2] = self.reward[b][2] + 1 * self.pot_weight 
			            else
			                self.reward[b][2] = self.reward[b][2] -1 * self.pot_weight
			            end
		            elseif a_t[b][2]==3 then
			            if a_t[b][1]==3 then
			                self.reward[b][2] = self.reward[b][2] + 1 * self.pot_weight
			            elseif self.agent_pos[b][1]<self.lever_pos[b][1] then
			                self.reward[b][2] = self.reward[b][2] - 3 * self.pot_weight
			            elseif self.agent_pos[b][1]>=self.lever_pos[b][1] then
			                self.reward[b][2] = self.reward[b][2] + 0.5 * self.pot_weight
			            end
		            end
	            end
            end

	        --cooperative reward for successfully pulling the lever
	        if self.terminal[b]==0 and (a_t[b][1] == 3 and a_t[b][2] == 3) then
		        earliest = torch.max(self.lever_pos[b])+1
		        if (self.step_counter == earliest) then
                    self.reward[b] = self.reward_all_live
		        end
		        self.terminal[b] = 1
	        elseif self.terminal[b]==0 and (a_t[b][1] == 3 or a_t[b][2] == 3) then
		        self.terminal[b] = 1
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
    local cAction = torch.zeros(self.opt.bs, self.opt.game_nagents):type(self.opt.dtype)
    
    for b = 1, self.opt.bs do
        for agent = 1, self.opt.game_nagents do
            if (step == 1) then --if on first step, stay in place so comm occurs
                pAction[b][agent] = 1 
            elseif (step <= self.opt.nsteps) then
                if (self.agent_pos[b][agent] < self.lever_pos[b][agent]) then --if agent is behind lever, move forward
                    pAction[b][agent] = 2 
                elseif (self.agent_pos[b][agent] > self.lever_pos[b][agent]) then --if agent is ahead of lever, stay
                    pAction[b][agent] = 1
                elseif (self.agent_pos[b][agent] == self.lever_pos[b][agent]) then --if agent at lever, functionality only for 2 agents
                    if (agent == 1) then
                        if (self.agent_pos[b][2] == self.lever_pos[b][2] and self.lever_pos[b][1] ~= 1 and self.lever_pos[b][2] ~= 1 ) then
                            pAction[b][agent] = 3 --pull if the other agent is at its lever too
                        else
                            pAction[b][agent] = 1 --stay put
                        end
                    elseif (agent == 2) then
                        if (self.agent_pos[b][1] == self.lever_pos[b][1] and self.lever_pos[b][1] ~= 1 and self.lever_pos[b][2] ~= 1) then
                            pAction[b][agent] = 3 --pull if the other agent is at its lever too
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
    
    if (self.opt.model_dial == 0) then
        for b = 1, self.opt.bs do
            for agent = 1, self.opt.game_nagents do
                if(self.lever_pos[b][agent] == 2) then
                    cAction[b][agent] = 4
                elseif (self.lever_pos[b][agent] == 3) then
                    cAction[b][agent] = 5
                elseif (self.lever_pos[b][agent] == 4) then
                    cAction[b][agent] = 6
                end
            end
        end
    end

    return cAction, pAction

end

return SimplePlan

