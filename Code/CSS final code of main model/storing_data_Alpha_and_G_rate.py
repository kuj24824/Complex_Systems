from multiprocessing import Pool, freeze_support
import time
import itertools
from multiprocessing import cpu_count
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

# "This file stores the grid data by using varing alpha and growth rate of commercial area. This is a second version of such a tuning"

# Constants
grid_size = 50
states = {'V': 0, 'H': 1, 'I': 2, 'C': 3}  # V: vacant, H: housing, I: industrial, C: commercial
growth_rates = {1: 0.003, 2: 0.001, 3: 0.0014}  
num_iterations = 30  # Number of iterations

# Initialize the grid and counts

counts = {0: grid_size * grid_size, 1: 0, 2: 0, 3: 0}

def initial_land_use(seed_number, grid_size, plot=True):
    np.random.seed(seed_number)
    
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # Define initial conditions
    center_x, center_y = grid_size // 2, grid_size // 2
    
    # Randomly select positions for commercial cells
    commercial_cells = []
    num_commercial = 3
    min_dist_commercial = 1  # Minimum distance from center
    
    while len(commercial_cells) < num_commercial:
        x = np.random.randint(center_x - min_dist_commercial, center_x + min_dist_commercial + 1)
        y = np.random.randint(center_y - min_dist_commercial, center_y + min_dist_commercial + 1)
        if (x, y) not in commercial_cells:
            commercial_cells.append((x, y))

    residential_cells = []
    num_residential = 25
    min_dist = 3  # Minimum distance from commercial cells

    industrial_cells = []
    num_industrial = 4
    min_dist_industrial = 3  # Minimum distance from commercial cells

    # Generate random residential cells around the commercial area
    while len(residential_cells) < num_residential:
        x = np.random.randint(center_x - min_dist, center_x + min_dist + 1)
        y = np.random.randint(center_y - min_dist, center_y + min_dist + 1)
        if (x, y) not in commercial_cells:
            residential_cells.append((x, y))

    # Generate random industrial cells around the commercial area
    while len(industrial_cells) < num_industrial:
        x = np.random.randint(center_x - min_dist_industrial, center_x + min_dist_industrial + 1)
        y = np.random.randint(center_y - min_dist_industrial, center_y + min_dist_industrial + 1)
        if (x, y) not in commercial_cells:
            industrial_cells.append((x, y))

    # Assign initial land use types
    for x, y in commercial_cells:
        grid[x, y] = states['C']  # Commercial
    for x, y in residential_cells:
        grid[x, y] = states['H']  # Residential
    for x, y in industrial_cells:
        grid[x, y] = states['I']  # Industrial

    if plot:
        # Set the figure size
        plt.figure(figsize=(8, 8))
        cmap = plt.cm.colors.ListedColormap(['white', 'skyblue', 'grey', 'orange'])
        plt.imshow(grid, cmap=cmap, origin='lower', vmin=0, vmax=3)
        plt.colorbar(ticks=[0, 1, 2, 3], label='Land Use')
        plt.title('Initial Land Use')
        plt.show()

    return grid


weights_table = {
    'Vacant_Commerce': {
        'C': [6, 3.5, 3, 2.5, 2, 2, 2, 1.5, 1.5, 1.5, 1.5, 1, 1, 1, 1, 1, 1, 1],
        'I': [0]*18,
        'H': [4, 3.5, 3, 2.5, 2, 2, 2, 1.5, 1.5, 1.5, 1.5, 1, 1, 1, 1, 1, 1, 1],
        'V': [0]*18
    },
    'Vacant_Industry': {
        'C': [0]*18,
        'I': [3, 3, 2, 1, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        'H': [-1, -1, 0] + [0]*15,
        'V': [0]*18
    },
    'Vacant_Housing': {
        'C': [-2, -1, 2, 1, 1, 1, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0, 0, 0, 0],
        'I': [-10, -10, -5, -3, -1] + [0]*13,
        'H': [2, 2, 1.5, 1.5, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
        'V': [0]*18
    },
    'Industry_Commerce': {
        'C': [25, 15, 10, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        'I': [-2, -2, -2] + [0]*15,
        'H': [4, 3.5, 3, 2.5, 2, 2, 2, 1.5, 1.5, 1.5, 1.5, 1, 1, 1, 1, 1, 1, 1],
        'V': [0]*18
    },
    'Industry_Industry': {
        'C': [0]*18,
        'I': [0]*18,
        'H': [0]*18,
        'V': [0]*18
    },
    'Industry_Housing': {
        'C': [0]*18,
        'I': [0]*18,
        'H': [0]*18,
        'V': [0]*18
    },
    'Housing_Commerce': {
        'C': [25, 15, 10, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        'I': [-10, -10, -5, -3, -1] + [0]*13,
        'H': [4, 3.5, 3, 2.5, 2, 2, 2, 1.5, 1.5, 1.5, 1.5, 1, 1, 1, 1, 1, 1, 1],
        'V': [0]*18
    },
    'Housing_Industry': {
        'C': [0]*18,
        'I': [3, 3, 2, 1, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        'H': [-1, -1, 0] + [0]*15,
        'V': [0]*18
    },
    'Housing_Housing': {
        'C': [0]*18,
        'I': [0]*18,
        'H': [0]*18,
        'V': [0]*18
    },
}



def get_distance_zone(distance):
    zone_mapping = {0: 1, 1: 1.4, 2: 2, 3: 2.2, 4: 2.8, 5: 3, 6: 3.2, 7: 3.6, 8: 4, 9: 4.1, 10: 4.2, 11: 4.5, 12: 5, 13: 5.1, 14: 5.4, 15: 5.7, 16: 5.8, 17: 6}
    for zone, max_distance in reversed(list(zone_mapping.items())):
        if distance >= max_distance:
            return zone
    return 0  # Return 0 if distance is less than the minimum specified distance

def get_neighbourhood(grid, row, col, radius):
    rows, cols = grid.shape
    square_row_range = range(max(0, row - radius), min(rows, row + radius + 1))
    square_col_range = range(max(0, col - radius), min(cols, col + radius + 1))
    square_neighbourhood = grid[np.ix_(square_row_range, square_col_range)]
    circle_mask = np.zeros_like(square_neighbourhood, dtype=bool)
    distance_zones = np.zeros_like(square_neighbourhood, dtype=int)
    # Adjusted center coordinates inside the neighbourhood
    center = min(row, radius), min(col, radius)
    for i in range(square_neighbourhood.shape[0]):
        for j in range(square_neighbourhood.shape[1]):
            distance = np.sqrt((center[0] - i) ** 2 + (center[1] - j) ** 2)
            if distance <= radius:
                circle_mask[i, j] = True
                distance_zones[i, j] = get_distance_zone(distance)
    circle_mask[center] = False  # Exclude the center cell
    return square_neighbourhood[circle_mask], distance_zones[circle_mask]




def cell_type_to_states(state):
    if state == 'Vacant':
        return 0
    elif state == 'Housing':
        return 1
    elif state == 'Industry':
        return 2
    elif state == 'Commerce':
        return 3

def states_to_cell_type(state):
    if state == 0:
        return 'V'
    elif state == 1:
        return 'H'
    elif state == 2:
        return 'I'
    elif state == 3:
        return 'C'


def calculate_transition_potentials_single_cell(args):
    i, j, grid, alpha, weights_table, radius = args
    transition_potentials = np.zeros(4)  # 4 possible states
    transitions = ['Vacant_Commerce', 'Vacant_Industry', 'Vacant_Housing', 'Industry_Commerce', 'Industry_Industry',
                   'Industry_Housing', 'Housing_Commerce', 'Housing_Industry', 'Housing_Housing']

    neighbourhood, distance_zones = get_neighbourhood(grid, i, j, radius)
    for transition in transitions:
        # Extract the current state and desired state from the transition string
        current_state, desired_state = transition.split('_')
        
        current_state_num = cell_type_to_states(current_state)
        desired_state_num = cell_type_to_states(desired_state)

        if grid[i, j] == current_state_num:
            sum_weights = 0
            for neighbor_state, distance_zone in zip(neighbourhood, distance_zones):
                neighbor_type = states_to_cell_type(neighbor_state)
                m_kd = weights_table[transition][neighbor_type][distance_zone]
                if neighbor_state == desired_state_num:
                    sum_weights += m_kd

            R = np.random.uniform(0, 1)
            S = 1 + (-math.log(R))**alpha
            transition_potentials[desired_state_num] = S * (1 + sum_weights)
    
    return ((i, j), transition_potentials)



def run_simulation(grid, weights_table, alpha, growth_rates, radius, seed, num_iterations, plot=False):
    np.random.seed(seed)
    grid_size = grid.shape[0]

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(grid_size):
        for j in range(grid_size):
            counts[grid[i, j]] += 1

    print(f"Initial counts: {counts}")

    for iteration in range(num_iterations):
        transition_potentials = np.zeros((grid.shape[0], grid.shape[1], 4))  # 4 possible states
        highest_potentials = {}
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i, j] == states['V']:
                    potential_states = ['H', 'I', 'C']
                elif grid[i, j] == states['H']:
                    potential_states = ['I', 'C']
                elif grid[i, j] == states['I']:
                    potential_states = ['C']
                else:
                    potential_states = []

                if potential_states:
                    highest_potentials[(i, j)] = max(potential_states,
                                                     key=lambda state: transition_potentials[i, j, states[state]])

        if __name__ == '__main__':
            pool = Pool()
            
            transition_potentials_dict = pool.map(calculate_transition_potentials_single_cell,
                                                  [(i, j, grid, alpha, weights_table, radius) for i in range(grid_size)
                                                   for j in range(grid_size)])
            pool.close()
            pool.join()

            for (i, j), potential in transition_potentials_dict:
                transition_potentials[i, j] = potential

        for new_state_key in sorted(states, key=lambda k: states[k], reverse=True):
            new_state = states[new_state_key]
            if new_state_key != 'V':
                num_to_convert = int(grid_size * grid_size * growth_rates[new_state_key])
                potential_cells = [(i, j) for i, j in highest_potentials.keys()
                                   if highest_potentials[(i, j)] == new_state_key]
                potential_cells.sort(key=lambda cell: transition_potentials[cell[0], cell[1],
                                                                            states[highest_potentials[cell]]],
                                     reverse=True)

                for cell in potential_cells[:num_to_convert]:
                    counts[grid[cell]] -= 1
                    counts[new_state] += 1
                    grid[cell] = new_state

        # Print the percentage of work done
        print(f"Percentage of work done: {(iteration + 1) / num_iterations * 100:.2f}%", end='\r')
    print(f"Final counts: {counts}")

    if plot:
        plt.figure(figsize=(10, 10))
        cmap = plt.cm.colors.ListedColormap(['white', 'skyblue', 'grey', 'orange'])
        plt.imshow(grid, cmap=cmap, origin='lower', vmin=0, vmax=3)
        plt.colorbar(ticks=[0, 1, 2, 3], label='Land Use')
        plt.title('Final Land Use')
        plt.show()

    return grid





if __name__ == '__main__':
    freeze_support()
    num_iterations = 50
    final_grids = {}
    alphas = [1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.125, 2.25, 2.375,
                    2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25, 3.375, 3.5, 3.625, 3.75, 3.875, 4.0]  # Example values for alpha
    grid_sizes = [75]  # Example values for grid size
    growth_rates_C = [0.0001, 0.0003, 0.0005, 0.0006, 0.0008, 0.001, 0.0012, 0.014, 0.016]  # Example values for growth rates of category 'C'

    for alpha in alphas:
        for grid_size in grid_sizes:
            for growth_rate_C in growth_rates_C:
                growth_rates = {
                    'H': 0.01,
                    'I': 0.002,
                    'C': growth_rate_C,
                }

                # Run the simulation
                start_time = time.time()
                grid = initial_land_use(seed_number=0, grid_size=grid_size, plot=False)
                final_grid = run_simulation(grid, weights_table, alpha=alpha, growth_rates=growth_rates, radius=6,
                                            seed=4567, num_iterations=num_iterations, plot=False)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Elapsed time for alpha={alpha}, grid_size={grid_size}, growth_rate_C={growth_rate_C}: {elapsed_time:.2f} seconds")

                # Store the final grid in the dictionary
                key = f"alpha_{alpha}_grid_{grid_size}_growth_rate_C_{growth_rate_C}"
                final_grids[key] = final_grid
    
    np.savez("final_grids2.npz", **final_grids)
    print("Final grids saved as final_grids2.npz")


