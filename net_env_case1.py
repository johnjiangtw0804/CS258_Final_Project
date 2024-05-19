import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import csv

objective_function_over_time = []
number_of_requests = 100

min_ht = 100
max_ht = 200

min_slots = 0
max_slots = 10

src_id = 1
dest_id = 7

class Request:
    def __init__(self, s, t, ht):
        self.s = s
        self.t = t
        self.ht = ht

class EdgeStat:
    def __init__(self, id, u, v, cap) -> None:
        self.id = int(id)
        self.u = u
        self.v = v
        self.cap = cap

        # spectrum state (a list of requests, showing color <-> request mapping). Each index of this list represents a color
        self.__slots = [None] * cap
        # a list of the remaining holding times corresponding to the mapped requests
        self.__hts = [0] * cap

    def add_request(self, req: Request, color:int):
        self.__slots[color] = req
        self.__hts[color] = req.ht
        self.cap -= 1

    def remove_requests(self):
        # iterate over the slots and decrement the holding time
        # this function is called at the end of each time slot, so when we decrement the holding time is 1, we remove the request
        for color, holding_time in enumerate(self.__hts):
            if holding_time > 1:
                self.__hts[color] -= 1
            elif holding_time == 1:
                self.__slots[color] = None
                self.cap += 1
                self.__hts[color] = 0


    def get_available_colors(self) -> list[int]:
        res = []
        for color, slot in enumerate(self.__slots):
            # if the slot is empty, it is available
            if slot is None:
                res.append(color)
        return res

class NetworkEnv(gym.Env):
    def __init__(self) -> None:
        self._G = nx.read_gml("data/nsfnet.gml")
        adj = nx.adjacency_matrix(self._G).todense()

        self._nodeID_to_name = {i: name for i, name in enumerate(self._G.nodes)}
        self._nodeName_to_ID = {name: i for i, name in enumerate(self._G.nodes)}

        self._adj_dict = {i: set() for i in range(len(self._G.nodes))}
        for i in range(len(self._G.nodes)):
            for j in range(len(self._G.nodes)):
                if adj[i, j] == 1:
                    self._adj_dict[i].add(j)
                if adj[j, i] == 1:
                    self._adj_dict[j].add(i)

        # define 2D array to store all the possible paths between two nodes
        self._possible_paths = [[[] for _ in range(len(self._G.nodes))] for _ in range(len(self._G.nodes))]
        for src in range(len(self._G.nodes)):
            for dest in range(len(self._G.nodes)):
                if src == dest:
                    continue
                # the default algorithm is dijkstra's algorithm
                shortest_path = nx.shortest_path(self._G, source=self._nodeID_to_name[src], target=self._nodeID_to_name[dest])
                allowed_additional_hops_path_len = len(shortest_path) + 2 # TODO: Might need to change it to a different number

                # define dfs function to calculate the possible paths between two nodes
                def dfs(graph, current, end, path):
                    if len(path) > allowed_additional_hops_path_len:
                        return
                    if (current == end):
                        self._possible_paths[src][dest].append(path.copy())
                        return
                    # find all the neighbors of the current node
                    for next in graph[current]:
                        # next not yet seen in the path
                        if next not in path:
                            path.append(next)
                            dfs(graph, next, end, path)
                            path.pop()
                dfs(self._adj_dict, src, dest, [src])

        self.observation_space = spaces.Dict(
            {
                "links": spaces.Box(min_slots, max_slots, shape=(len(self._G.edges),), dtype=int),
                "req_ht": spaces.Discrete(max_ht+1),
            }
        )

        self.blocking_action = len(self._possible_paths[src_id][dest_id])
        self.action_space = spaces.Discrete(len(self._possible_paths[src_id][dest_id]) + 1)
        self.round = 0

        self._round_to_EdgeStats = []
        EdgeStats = [] # EdgeStae * len(edges)
        for u, v in self._G.edges:
            # id get
            if self._G[u][v].get('id') is None:
                id = 3
            else:
                id = self._G[u][v]['id'][1:]
            u, v = self._nodeName_to_ID[u], self._nodeName_to_ID[v]
            estat = EdgeStat(id, u, v, max_slots)
            EdgeStats.append(estat)
        self._round_to_EdgeStats.append(EdgeStats) # for round zero

    def _generate_req(self):
        return np.array([Request(src_id, dest_id, np.random.randint(min_ht, max_ht))])

    def _get_obs(self):
        return {
            "links": self._linkstates,
            "req_ht": self._req[0].ht,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._linkstates = np.array([0] * self._G.number_of_edges())

        self._round_to_EdgeStats = []
        EdgeStats = []
        for u, v in self._G.edges:
            if self._G[u][v].get('id') is None:
                # there is no id attribute, so we assign a missing value
                id = 3
            else:
                id = self._G[u][v]['id'][1:]
            u, v = self._nodeName_to_ID[u], self._nodeName_to_ID[v]
            estat = EdgeStat(id, u, v, max_slots)
            EdgeStats.append(estat)
        self._round_to_EdgeStats.append(EdgeStats) # for round zero

        self._req = self._generate_req()
        observation = self._get_obs()
        info = {}
        self.round = 0
        return observation, info

    def step(self, action):
        # rewards
        blocking_reward = -1
        success_reward = 1

        # if the action is blocking or non-reachable, we dont change the link state, so we make a copy of it
        old_linkstates = self._linkstates.copy()
        old_edgeStats = self._round_to_EdgeStats[self.round]

        if action == self.blocking_action:
            reward = blocking_reward
        else: # routing the request
            path = self._possible_paths[src_id][dest_id][action]
            # it is reachable, so we try to route the request
            def route(path, edgestats, linkstates, request) -> tuple[bool, list[EdgeStat], np.array]:
                for i in range(len(path) - 1):
                    # 0 1 2 3
                    u = path[i]
                    v = path[i+1]
                    # check if the link has available slots
                    for e in edgestats:
                        if e.u == u and e.v == v:
                            available_colors = e.get_available_colors()
                            # blocking
                            if len(available_colors) == 0:
                                return False, edgestats, linkstates
                            # update the edge state
                            e.add_request(request, e.get_available_colors()[0])
                            # update the link state
                            linkstates[e.id] += 1
                            break
                return True, edgestats, linkstates

            isOK, new_edgeStats, new_linkStates = route(path, old_edgeStats, old_linkstates, self._req[0])
            if  isOK:
                reward = success_reward
            # this could happen if the path is blocked
            else:
                reward = blocking_reward

        self.round += 1
        if reward == blocking_reward:
            self._round_to_EdgeStats.append(old_edgeStats)
            self._linkstates = old_linkstates
        else:
            self._round_to_EdgeStats.append(new_edgeStats)
            self._linkstates = new_linkStates

        # remove requests that the holding time is 1
        for e in self._round_to_EdgeStats[self.round]:
            prev_cap = e.cap
            e.remove_requests()
            # update the link state if the capacity has changed
            if prev_cap != e.cap:
                self._linkstates[e.id] -= e.cap - prev_cap

        # next round
        self._req = self._generate_req()
        observation = self._get_obs()
        info = {}
        terminated = (self.round  == number_of_requests)
        util_t_e = 0.0

        # objective over this episode
        if terminated:
            for EdgeStats in self._round_to_EdgeStats:
                for estat in EdgeStats:
                    util_t_e += (max_slots - estat.cap) / max_slots # utilization of the link
            util_t_e = util_t_e / (number_of_requests * len(EdgeStats))
            # (the objective vs episode)
            objective_function_over_time.append(util_t_e)
            data = [util_t_e]

            # Append to the CSV file
            with open('objective_over_episode.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                # Write the float as a row to the CSV file
                writer.writerow(data)
        return observation, reward, terminated, terminated, info