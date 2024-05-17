import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

number_of_requests = 100

min_ht = 10
max_ht = 20

min_slots = 0
max_slots = 10

class Request:
    def __init__(self, s, t, ht):
        self.s = s
        self.t = t
        self.ht = ht

class EdgeStat:
    def __init__(self, id, u, v, cap) -> None:
        self.id = id
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
            if holding_time == 1:
                self.__slots[color] = None
                self.cap += 1
                continue
            self.__hts[color] -= 1

    def get_available_colors(self) -> list[int]:
        res = []
        for color, slot in enumerate(self.__slots):
            # if the slot is empty, it is available
            if slot is None:
                res.append(color)
        return res

    # utilization of the link
    def show_spectrum_state(self) -> float:
        occupied_counter = 0
        for _, slot in enumerate(self.__slots):
            if slot is not None:
                occupied_counter += 1
        return occupied_counter / len(self.__slots)

class NetworkEnv(gym.Env):
    # private variables
    def __init__(self) -> None:
        self._G = nx.read_gml("data/nsfnet.gml")
        adj = nx.adjacency_matrix(self._G).todense()

        # we need these attributes to convert the node name to node ID and vice versa
        self._nodeID_to_name = {i: name for i, name in enumerate(self._G.nodes)}
        self._nodeName_to_ID = {name: i for i, name in enumerate(self._G.nodes)}

        # Personally, I like to use dictionary to store the adjacency list
        self._adj_dict = {i: set() for i in range(len(self._G.nodes))}
        for i in range(len(self._G.nodes)):
            for j in range(len(self._G.nodes)):
                if adj[i, j] == 1:
                    self._adj_dict[i].add(j)
                if adj[j, i] == 1:
                    self._adj_dict[j].add(i)

        # define 2D array to store all the possible paths between two nodes
        # the path length is less than or equal to the shortest path length + 3
        self._possible_paths = [[[] for _ in range(len(self._G.nodes))] for _ in range(len(self._G.nodes))]
        for src in self._adj_dict:
            for dest in self._adj_dict[src]:
                # the default algorithm is dijkstra's algorithm
                shortest_path = nx.shortest_path(self._G, source=self._nodeID_to_name[src], target=self._nodeID_to_name[dest])
                allowed_additional_hops_path_len = len(shortest_path) + 3 # TODO: Might need to change it to a different number

                # define dfs function to calculate the possible paths between two nodes
                def dfs(graph, current, end, path):
                    if len(path) > allowed_additional_hops_path_len:
                        return
                    if (current == end):
                        self._possible_paths[src][dest].append(path.copy())
                        return
                    for next in graph[current]:
                        path.append(next)
                        dfs(graph, next, end, path)
                        path.pop()
                dfs(self._adj_dict, src, dest, [src])

        # action(int) to path (list) converter
        self._action_to_path = []
        total_paths = 0
        for i in range(len(self._G.nodes)):
            for j in range(len(self._G.nodes)):
                if i != j:
                    total_paths += len(self._possible_paths[i][j])
                    for path in self._possible_paths[i][j]:
                        self._action_to_path.append((path))

        self.observation_space = spaces.Dict(
            {
                "links": spaces.Box(min_slots, max_slots, shape=(len(self._G.edges),), dtype=int),
                "req_ht": spaces.Discrete(max_ht+1), # Discrete is exclusive
                "req_src": spaces.Discrete(len(self._G.nodes)), # I believe src and dest are useful for the agent to learn
                "req_dst": spaces.Discrete(len(self._G.nodes)),
            }
        )

        # assuming #(total_paths) is the blocking action
        self.action_space = spaces.Discrete(total_paths + 1) # +1 since discrete is exclusive and we have a blocking action
        self.blocking_action = total_paths
        self.round = 0

        # 這是我們可以用來存 edge state for every t，然後用它來算 objective function
        self._round_to_EdgeStats = []
        EdgeStats = []
        for u, v in self._G.edges:
            if self._G[u][v].get('id') is None:
                id = 3
            else:
                id = self._G[u][v]['id'][1:]
            estat = EdgeStat(id, u, v, max_slots)
            EdgeStats.append(estat)
        self._round_to_EdgeStats.append(EdgeStats) # for round zero

    def _generate_req(self):
        # case 1
        src = self._nodeName_to_ID["San Diego Supercomputer Center"]
        dest = self._nodeName_to_ID["Jon Von Neumann Center, Princeton, NJ"]
        return np.array([Request(src, dest, np.random.randint(min_ht, max_ht))])

    def _get_obs(self):
        return {
            "links": self._linkstates,
            "req_ht": self._req[0].ht,
            "req_src": self._req[0].s,
            "req_dst": self._req[0].t,
        }

    def reset(self, seed=None, options=None):
        debug = self
        super().reset(seed=seed)
        self._linkstates = np.array([0] * self._G.number_of_edges())

        self._round_to_EdgeStats = []
        EdgeStats = []
        for u, v in self._G.edges:
            if self._G[u][v].get('id') is None:
                id = 3
            else:
                id = self._G[u][v]['id'][1:]
            estat = EdgeStat(id, u, v, max_slots)
            EdgeStats.append(estat)
        self._round_to_EdgeStats.append(EdgeStats) # for round zero
        self._req = self._generate_req()
        observation = self._get_obs()
        info = {}
        self.round = 0
        return observation, info

    def step(self, action):
        debug = self
        # rewards
        blocking_reward = -1
        non_reachability_penalty = -5 # I think this should be a large negative number. I am not sure
        success_reward = 1

        # if the action is blocking or non-reachable, we dont change the link state, so we make a copy of it
        old_linkstates = self._linkstates.copy()
        old_edgeStats = self._round_to_EdgeStats[self.round]

        if action == self.blocking_action:
            reward = blocking_reward
        else: # routing the request
            path = self._action_to_path[action]
            src = path[0]
            dest = path[-1]

            # early return if the path is not reachable
            if src != self._req[0].s or dest != self._req[0].t:
                reward = non_reachability_penalty
            else:
                def route(path, edgestats, linkstates, request) -> tuple[bool, list[EdgeStat], np.array]:
                    for i in range(len(path) - 1):
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
        if reward == non_reachability_penalty or reward == blocking_reward:
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
                self._linkstates[e.id] += e.cap - prev_cap

        ROUND_TO_EDGE_STATS = self._round_to_EdgeStats

        # next round
        self._req = self._generate_req()
        observation = self._get_obs()
        info = {}
        terminated = (self.round  == number_of_requests)
        return observation, reward, terminated, terminated, info


       # Network-wide utilization
        # https://github.com/sjsu-interconnect/cs258/blob/main/projects/rsa.md
        # 1. The utilization at time t for link e (ute) = occupied slots at time t / total slots of link e
        # 2. The average utilization of link e over T episodes = (Σ ute over T) / T.
        # 3. The formal objective function is to achieve maximum network-wide utilization. We define the network-wide utilization
        #    as the average of the edge total utility:
        #    Σ (Σ ute over T / T) over all edges e in the network



