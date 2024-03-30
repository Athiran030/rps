import traci
import numpy as np
import timeit
import random

class Mysimulation:
    
    def __init__(self, Model, Memory, gamma, TrafficGen, sumo_cmd, max_steps, num_states, num_actions, training_epochs):
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._vehicleId = None
        self._step = 0
        self._num_states = num_states
        self._num_actions = num_actions
        self._Model = Model
        self._Memory = Memory
        self._gamma = gamma
        self._training_epochs = training_epochs
        self._reward_store = []
        self._cumulative_wait_store = []
        self._cumulative_cost_store = []

    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        #self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._sum_reward = 0 #car will get a reward at each cs, for the whole episode this will store the tot. reward
        self._sum_waiting_time = 0
        self._sum_cost = 0
        self._travel_time = 0
        self._at_charging_station = True
        self._rl_step = 0
        vid = 'v_1'
        old_state = []
        old_action = []

        #Adding the vehicle of our interest into the sumo environment
        traci.vehicle.add(vid, routeID='r_0')

        while self._step < self._max_steps:
            #generating samples
            #training the samples
            
            if self._at_charging_station:
                current_state = self._get_state()

                #choosing the action, whether to stop or not at the current charging station
                action = self._choose_action(current_state, epsilon)

                self._simulate(1)

                reward = self._calculate_reward()

                #adding the sample into the memory
                self._Memory.add_sample((old_state, old_action, reward, current_state))
            else:
                self._simulate(1)

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Simulates the waiting and charging time of the EV vehichle
        Car will have Blue in wait, and Yellow during charge
        """
        traci.simulationStep()
        self._step += 1

    def _get_state(self):
        """
        Retrieve the information of charging stations and EV from SUMO
        """
        state = []
        charging_Station_list = traci.chargingstation.getIDList()

        for charging_station_id in charging_Station_list:
            waiting_time = traci.chargingstation.getParameter(charging_station_id, 'waiting_time')
            cost = traci.chargingstation.getParameter(charging_station_id, 'cost')
            charging_time = traci.chargingstation.getParameter(charging_station_id, 'charging_time')

            state.append([waiting_time, cost, charging_time])
        
        return np.array(state)


    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state

    def _calculate_reward(self):
        return 0

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._cumulative_cost_store.append(self._sum_cost)  # average number of queued cars per step, in this episode


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def cumulative_cost_store(self):
        return self._avg_queue_length_store

