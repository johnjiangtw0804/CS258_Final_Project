import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

class NetworkEnv(gym.Env):
    # private variables
    def __init__(self) -> None:
        G = nx.read_gml("data/nsfnet.gml")
        adj = nx.adjacency_matrix(G).todense()

        # Personally, like to use dict to store the adjacency list
        self._adj_dict = {i: set() for i in range(len(G.nodes))}
        for i in range(len(G.nodes)):
            for j in range(len(G.nodes)):
                if adj[i, j] == 1:
                    self._adj_dict[i].add(j)
                if adj[j, i] == 1:
                    self._adj_dict[j].add(i)

        # we need to store the nodeID to name mapping because dijkstra's algorithm uses node name
        self._nodeID_to_name = {i: name for i, name in enumerate(G.nodes)}

        # define 2d array to store the possible paths between two nodes
        self._possible_paths = [[[] for _ in range(len(G.nodes))] for _ in range(len(G.nodes))]
        for src in self._adj_dict:
            for dest in self._adj_dict[src]:
                shortest_path = nx.shortest_path(G, source=self._nodeID_to_name[src], target=self._nodeID_to_name[dest])
                allowed_additional_hops_path_len = len(shortest_path) + 3

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

        # test cases I used to verify the correctness of the code
        # [[0, 2], [0, 11, 0, 2], [0, 7, 0, 2]] <= possible paths between node 0 and node 2. I tested with my eyes
        # print(self._possible_paths[0][2])

        # number of links
        # print(len(G.edges))
        self.observation_space = spaces.Dict(
            {
                # need to define the range of the number of occupied slots
                # for us, it would be 10 for each link and shape would be the # of links
                "links": spaces.Box(0, 10, shape=(len(G.edges),), dtype=int),
                "req": spaces.Box(10, 20, shape=(100,), dtype=int)
            }
        )
        self.action_space = spaces.Discrete()

        self.round = 0

    def _generate_req(self):
        # a request (indicating the capacity required to host this request)
        return np.array([np.random.randint(1, 3+1),]) # holding time, Source, Destination

    def _get_obs(self):
        return {
            "links": self._linkstates,
            "req": self._req,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # to recover the original state (empty)
        self._linkstates = np.array([0] * 3)

        # to generate a request
        self._req = self._generate_req()

        observation = self._get_obs()
        info = {}

        self.round = 0

        return observation, info
    # S [a,b] b = list()

    def step(self, action):
        # action = 0 (P0), 1 (P1), 2 (P2), 3 (blocking) # shortest path +=3
        # if we have enough capacity to host the req on the selected path
        # self._req[0]: requested capacity
        blocking_action = 3
        blocking_reward = -1

        self.round += 1
        terminated = (self.round == 8) # True if it experienced 8 rounds

        if action == blocking_action:
            # we need to block
            reward = blocking_reward
        else: # routing
            num_occupied_slots = self._linkstates[action] + self._req[0] # 3
            if num_occupied_slots <= 5: # we can map
                self._linkstates[action] = num_occupied_slots
                reward = +1 * self._req[0]
            else: # we need to block
                reward = blocking_reward

        self._req = self._generate_req()
        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, terminated, info

