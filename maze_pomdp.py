import random
import pickle
import itertools
import pomdp_py
import numpy as np
import matplotlib.pyplot as plt
from layered_maze_lib import maze_composer
from process_human_data import get_test_stimuli


MAZE_SIZE = 20
MAZE_DIR = 'layered_maze_lib/path_datasets/maze_size_{}/samples_per_pair_100_v0'.format(MAZE_SIZE)
NUM_LAYERS = 50
PIXELS_PER_SQUARE = 2
IMG_DIM = PIXELS_PER_SQUARE * MAZE_SIZE - 1

FOVEA_SIZE = 3
N_SACCADES = 6

CURR_MAZE = np.zeros((IMG_DIM, IMG_DIM)) - 1e9
CURR_EXIT = np.zeros(2) - 1e9
TRANSITION = False

RAND_SEED = 1033

POSSIBLE_FOVEA_WINDOWS = [np.array(seq).reshape(FOVEA_SIZE, FOVEA_SIZE) for seq in itertools.product([0, 1], repeat=FOVEA_SIZE**2)]


test_mazes, test_entrances, test_exits, test_paths = get_test_stimuli()


def get_xy(str_rep):
    return np.array([int(str_rep[1:3]), int(str_rep[5:7])])


def rotate(points, theta, xo=IMG_DIM / 2 - 0.5, yo=IMG_DIM / 2 - 0.5):
    """Rotate points around xo,yo by theta (rad) clockwise."""
    xr = np.cos(theta) * (points[:, 0] - xo) - np.sin(theta) * (points[:, 1] - yo) + xo
    yr = np.sin(theta) * (points[:, 0] - xo) + np.cos(theta) * (points[:, 1] - yo) + yo
    return np.dstack((xr, yr))[0]


class FoveaState(pomdp_py.State):
    def __init__(self, data, x, y):
        self.x = x
        self.y = y
        self.name = str(data)
        self.data = data
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, FoveaState):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "FoveaState(%s)" % self.name


class SaccadeAction(pomdp_py.Action):
    """
    The agent can jump to any integer coordinate on the maze.
    Represented by '(xx, yy)'.
    """
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, SaccadeAction):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "SaccadeAction(%s)" % self.name


class FoveaObservation(pomdp_py.Observation):
    def __init__(self, data):
        self.name = str(data)
        self.data = data
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, FoveaObservation):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "FoveaObservation(%s)" % self.name


def get_fovea_window(maze, x, y):
    padded_maze = np.pad(maze, FOVEA_SIZE//2, constant_values=1)
    return padded_maze[y:y+2*(FOVEA_SIZE//2)+1, x:x+2*(FOVEA_SIZE//2)+1]


class ObservationModel(pomdp_py.ObservationModel):
    def sample(self, next_state, action):
        # based on next state location in maze, sample an observation
        fovea_window = get_fovea_window(CURR_MAZE, next_state.x, next_state.y)

        def flip_bit(arr, x, y):
            arr[x][y] = 1 - arr[x][y]
            return arr

        fovea_window = flip_bit(fovea_window, np.random.randint(FOVEA_SIZE), np.random.randint(FOVEA_SIZE))
        fovea_window = flip_bit(fovea_window, np.random.randint(FOVEA_SIZE), np.random.randint(FOVEA_SIZE))
        fovea_window = flip_bit(fovea_window, np.random.randint(FOVEA_SIZE), np.random.randint(FOVEA_SIZE))

        return FoveaObservation(fovea_window)

    def probability(self, observation, next_state, action):
        return 1/165 if np.sum(np.abs(next_state.data - observation.data)) <= 3 else 1e-9

    def get_all_observations(self):
        possible_obs = [FoveaObservation(arr) for arr in POSSIBLE_FOVEA_WINDOWS]
        return possible_obs


class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        if TRANSITION:
            return 0.5
        return 1-1e-9 if next_state.name == state.name else 1e-9

    def sample(self, state, action):
        return FoveaState(get_fovea_window(CURR_MAZE, state.x, state.y), state.x, state.y)

    # def get_all_states(self):
    #     possible_states = [FoveaState(arr) for arr in POSSIBLE_FOVEA_WINDOWS]
    #     return possible_states


class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        curr_pos = get_xy(action.name)
        distance = np.linalg.norm(curr_pos - CURR_EXIT)
        # if distance > 20:
        #     return -200
        if np.linalg.norm(np.array([state.x, state.y]) - curr_pos) > 15:
            return -1000
        max_dist = np.sqrt(2*IMG_DIM**2)
        max_reward = 200
        return max_reward/(max_dist**4) * (distance - max_dist) ** 4 - 3*np.linalg.norm(np.array([state.x, state.y]) - curr_pos)

    def sample(self, state, action, next_state):
        return self._reward_func(state, action)


class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a small, finite action space"""
    ACTIONS = {SaccadeAction('({}, {})'.format(str(x).zfill(2), str(y).zfill(2))) for x,y in np.transpose(np.meshgrid(np.arange(39), np.arange(39)), (1, 2, 0)).reshape(-1, 2)}

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS


class MazeProblem(pomdp_py.POMDP):
    def __init__(self, x, y, init_true_state, init_belief):
        """init_belief is a Distribution."""
        self.x = x
        self.y = y
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(),
                               RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name='MazeProblem')


def generate_maze():
    composer = maze_composer.MazeComposer(path_dir=MAZE_DIR, num_layers=NUM_LAYERS, pixels_per_square=PIXELS_PER_SQUARE)
    maze, path = composer()
    path = np.flip(path, axis=1)
    n_rot = np.random.randint(4)
    maze = np.rot90(maze, k=n_rot)
    theta = (-n_rot * np.pi / 2) % (2 * np.pi)
    path = rotate(path, theta)
    entrance = path[0]
    exit_point = path[-1]
    maze = 1 - maze
    return maze, entrance, exit_point


def train(maze_problems, planners, n_mazes=100):
    eye_paths = []
    train_mazes = []
    train_entrances = []
    train_exits = []

    for i in range(n_mazes):
        print('============ Maze {} ============'.format(i+1))

        maze, entrance, exit_point = generate_maze()
        train_mazes.append(maze)
        train_entrances.append(entrance)
        train_exits.append(exit_point)

        global CURR_MAZE, CURR_EXIT, TRANSITION
        CURR_MAZE = train_mazes[i]
        CURR_EXIT = train_exits[i]

        # update all grid models

        curr_x, curr_y = int(train_entrances[i][0]), int(train_entrances[i][1])
        eye_path = [[curr_x, curr_y]]
        for j in range(N_SACCADES):
            print('----- Step {} -----'.format(j))
            print('({:.0f}, {:.0f})'.format(curr_x, curr_y))

            curr_maze_problem = maze_problems[curr_x][curr_y]
            curr_planner = planners[curr_x][curr_y]
            print("True state:\n", curr_maze_problem.env.state)
            # print("Belief:\n", curr_maze_problem.agent.cur_belief)

            action = curr_planner.plan(curr_maze_problem.agent)
            new_x, new_y = get_xy(action.name)
            eye_path.append([new_x, new_y])
            print("Action:", action)

            TRANSITION = j == N_SACCADES-1
            reward = curr_maze_problem.env.state_transition(action, execute=True)
            ## reward = curr_maze_problem.env.reward_model.sample(curr_maze_problem.env.state, action, None)
            print("Reward:", reward)

            real_observation = curr_maze_problem.agent.observation_model.sample(curr_maze_problem.env.state, action)
            print(">> Observation:\n", real_observation)

            curr_maze_problem.agent.update_history(action, real_observation)
            curr_planner.update(curr_maze_problem.agent, action, real_observation)

            if isinstance(curr_maze_problem.agent.cur_belief, pomdp_py.Histogram):
                new_belief = pomdp_py.update_histogram_belief(
                    curr_maze_problem.agent.cur_belief,
                    action, real_observation,
                    curr_maze_problem.agent.observation_model,
                    curr_maze_problem.agent.transition_model)
                curr_maze_problem.agent.set_belief(new_belief)

            curr_x, curr_y = new_x, new_y

        eye_paths.append(np.array(eye_path))
    return eye_paths, train_mazes, train_exits


def test(maze_problems, planners):
    eye_paths = []
    for i in range(50):
        print('============ Maze {} ============'.format(i+1))

        global CURR_MAZE, CURR_EXIT, TRANSITION
        CURR_MAZE = test_mazes[i]
        CURR_EXIT = test_exits[i]

        # update all grid models

        curr_x, curr_y = int(test_entrances[i][0]), int(test_entrances[i][1])
        eye_path = [[curr_x, curr_y]]
        for j in range(N_SACCADES):
            print('----- Step {} -----'.format(j))
            print('({:.0f}, {:.0f})'.format(curr_x, curr_y))

            curr_maze_problem = maze_problems[curr_x][curr_y]
            curr_planner = planners[curr_x][curr_y]
            print("True state:\n", curr_maze_problem.env.state)
            # print("Belief:\n", curr_maze_problem.agent.cur_belief)

            action = curr_planner.plan(curr_maze_problem.agent)
            new_x, new_y = get_xy(action.name)
            eye_path.append([new_x, new_y])
            print("Action:", action)

            TRANSITION = j == N_SACCADES-1
            reward = curr_maze_problem.env.state_transition(action, execute=True)
            ## reward = curr_maze_problem.env.reward_model.sample(curr_maze_problem.env.state, action, None)
            print("Reward:", reward)

            real_observation = curr_maze_problem.agent.observation_model.sample(curr_maze_problem.env.state, action)
            print(">> Observation:\n", real_observation)

            curr_maze_problem.agent.update_history(action, real_observation)
            curr_planner.update(curr_maze_problem.agent, action, real_observation)

            if isinstance(curr_maze_problem.agent.cur_belief, pomdp_py.Histogram):
                new_belief = pomdp_py.update_histogram_belief(
                    curr_maze_problem.agent.cur_belief,
                    action, real_observation,
                    curr_maze_problem.agent.observation_model,
                    curr_maze_problem.agent.transition_model)
                curr_maze_problem.agent.set_belief(new_belief)

            curr_x, curr_y = new_x, new_y

        eye_paths.append(np.array(eye_path))
    return eye_paths


########################################################################################################################

np.random.seed(RAND_SEED)
first_maze, first_entrance, first_exit_point = generate_maze()

init_beliefs = [[pomdp_py.Histogram({FoveaState(arr, x, y):1/len(POSSIBLE_FOVEA_WINDOWS) for arr in POSSIBLE_FOVEA_WINDOWS}) for y in range(IMG_DIM)] for x in range(IMG_DIM)]

maze_problems = [[MazeProblem(x, y,
                              FoveaState(get_fovea_window(first_maze, x, y), x, y),
                              init_beliefs[x][y])
                  for y in range(IMG_DIM)] for x in range(IMG_DIM)]

planners = np.array([[pomdp_py.POUCT(max_depth=3, discount_factor=0.95,
                       num_sims=4096, exploration_const=10,
                       rollout_policy=maze_problems[x][y].agent.policy_model,
                       show_progress=False) for y in range(IMG_DIM)] for x in range(IMG_DIM)])

np.random.seed(RAND_SEED)
# train_eye_paths, train_mazes, train_exits = train(maze_problems, planners, n_mazes=4)
test_eye_paths = test(maze_problems, planners)

with open('test_eye_paths.pickle', 'wb') as f:
    pickle.dump(test_eye_paths, f)


def plot_trial(idx):
    plt.imshow(test_mazes[idx])
    plt.plot(*test_exits[idx].T, marker='o', ms=12, c='g')
    plt.plot(*test_eye_paths[idx].T, marker='o')


