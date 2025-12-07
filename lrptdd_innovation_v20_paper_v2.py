"""
LRPTDD Fully Compliant Code (v22 - Output Reconstruction)
---------------------------------------------------------
- FIX: Introduces get_full_multi_sequence_routes to re-run TDD evaluation and capture ALL necessary routes (x1, x2, ...) required to satisfy demand, finally confirming multi-visit logic visually.
"""

import pandas as pd
import random
import math
import time
import matplotlib.pyplot as plt
import csv
import os

# --- Configuration (Keep consistent with previous runs) ---
# NOTE: The file access is now configured directly in the main execution block
# to use the uploaded file names, which are assumed to be in the current directory.

# Since no BASE_PATH is defined when executing in a notebook/VM, we remove it or ensure it's handled.
# For simplicity, we assume the CSVs are in the same folder as the script.

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    "customer_file": os.path.join(BASE_PATH, "data", "customers.csv"),
    "depot_file": os.path.join(BASE_PATH, "data", "depots.csv"),
    "vehicle_capacity": 50,
    "vehicle_cost": 500,
    "sa_params": {
        "T0": 30,
        "Tf": 0.01,  # origin 0.01
        "alpha": 0.98,
        "K": 1 / 9,
        "iter_factor": 6000,  # origin 6000
        "I_iter_factor": 100,  # Inner loop iterations # origin 100 <-- ADDED for inner loop iterations
        "N_non_improving": 150,  # Max non-improving reductions  # origin 150 # <-- ADDED for non-improving counter
    },
    "local_search_steps": 100,
    "pso_params": {
        "swarm_size": 5,
        "max_iter": 50,
        "w": 0.5,
        "c1": 2,
        "c2": 2,
        "sa_interval": 10,
        "elite_k": 2,
    },
}

# --- Data Loading (Assuming files are in the same directory) ---
customers_df = pd.read_csv(CONFIG["customer_file"], delimiter=";")
depots_df = pd.read_csv(CONFIG["depot_file"], delimiter=";")

customers_df["id"] = customers_df["id"].astype(int)
depots_df["id"] = depots_df["id"].astype(int)

VEHICLE_CAPACITY = CONFIG["vehicle_capacity"]
VEHICLE_COST = CONFIG["vehicle_cost"]
CUSTOMER_IDS = customers_df["id"].tolist()
DEPOT_IDS = depots_df["id"].tolist()
EPSILON = 1e-4

# ------------------------------------------------------------------
#                   I. CORE UTILITY FUNCTIONS
# ------------------------------------------------------------------


def get_coordinates(node_id):
    node_id = int(node_id)
    if node_id in customers_df["id"].values:
        row = customers_df[customers_df["id"] == node_id].iloc[0]
        return row["x"], row["y"]
    elif node_id in depots_df["id"].values:
        row = depots_df[depots_df["id"] == node_id].iloc[0]
        return row["x"], row["y"]
    else:
        return 0, 0


def calculate_distance(node1, node2):
    x1, y1 = get_coordinates(node1)
    x2, y2 = get_coordinates(node2)
    return math.dist((x1, y1), (x2, y2))


def calculate_tour_cost(tour):
    cost = 0
    if len(tour) < 2:
        return 0
    for i in range(len(tour) - 1):
        cost += calculate_distance(tour[i], tour[i + 1])
    return cost


def two_opt_swap_list(tour, i, k):
    new_tour = tour[:i] + tour[i : k + 1][::-1] + tour[k + 1 :]
    return new_tour


def run_two_opt(nodes, max_iterations=50):
    if len(nodes) < 4:
        return nodes
    best_tour = nodes
    best_cost = calculate_tour_cost(best_tour)
    for _ in range(max_iterations):
        improved = False
        for i in range(1, len(best_tour) - 2):
            for k in range(i + 1, len(best_tour) - 1):
                new_tour = two_opt_swap_list(best_tour, i, k)
                new_cost = calculate_tour_cost(new_tour)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_tour = new_tour
                    improved = True
        if not improved:
            break
    return best_tour


def get_customer_demand(cust_id):
    if cust_id in customers_df["id"].values:
        return customers_df[customers_df["id"] == cust_id].iloc[0]["demand_total"]
    return 0


def capacity_split(depot_id, tour):
    routes = []
    current_route = [depot_id]
    current_load = 0
    customers_in_order = tour[1:-1]
    for customer in customers_in_order:
        demand = get_customer_demand(customer)
        if current_load + demand <= VEHICLE_CAPACITY + EPSILON:
            current_route.append(customer)
            current_load += demand
        else:
            current_route.append(depot_id)
            routes.append(current_route)
            current_route = [depot_id, customer]
            current_load = demand
    if len(current_route) > 1:
        current_route.append(depot_id)
        routes.append(current_route)
    return routes


def feasibility_check(routes):
    return True


def apply_swap(solution):
    if len(solution) < 2:
        return solution
    i, j = random.sample(range(len(solution)), 2)
    new_solution = solution[:]
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution


def apply_insertion(solution):
    if len(solution) < 2:
        return solution
    new_solution = solution[:]
    i, j = random.sample(range(len(solution)), 2)
    element = new_solution.pop(i)
    new_solution.insert(j, element)
    return new_solution


def apply_2opt_move(solution):
    if len(solution) < 2:
        return solution
    new_solution = solution[:]
    i, k = random.sample(range(len(solution)), 2)
    i, k = min(i, k), max(i, k)
    new_solution[i : k + 1] = new_solution[i : k + 1][::-1]
    return new_solution


def get_neighbor(solution, move_type=None):
    if move_type is None:
        move_type = random.randint(1, 3)
    if move_type == 1:
        return apply_swap(solution), "Swap"
    elif move_type == 2:
        return apply_insertion(solution), "Insertion"
    elif move_type == 3:
        return apply_2opt_move(solution), "2-opt"
    return solution, "None"


# ------------------------------------------------------------------
#                   II. CORE LRPTDD CLASSES & LOGIC
# --------------------------------------------------


class GlobalCustomerTracker:
    def __init__(self, state=None, costs=None):
        if state is None:
            self.state = {}
            for idx, row in customers_df.iterrows():
                cust_id = int(row["id"])
                self.state[cust_id] = {
                    "remaining_demand": row["demand_total"],
                    "last_pickup_time": row["start_time"],
                    "demand_rate": row["demand_rate"],
                }
            self.cost_records = []
        else:
            self.state = {k: v.copy() for k, v in state.items()}
            self.cost_records = list(costs)

    def get_remaining_demand(self, cust_id):
        return self.state[cust_id]["remaining_demand"]

    def update_pickup(self, cust_id, picked_load, pickup_time):
        self.state[cust_id]["remaining_demand"] = max(
            0, self.state[cust_id]["remaining_demand"] - picked_load
        )
        self.state[cust_id]["last_pickup_time"] = pickup_time

    def get_uncollected_customers(self):
        return [
            cid for cid in CUSTOMER_IDS if self.state[cid]["remaining_demand"] > EPSILON
        ]

    def get_current_state(self):
        return self.state, self.cost_records

    def get_total_cost(self):
        return sum(self.cost_records)


def compute_arrival_times(route):
    if len(route) < 2:
        return [0]
    times = [0]
    current_time = 0
    for i in range(len(route) - 1):
        x1, y1 = get_coordinates(route[i])
        x2, y2 = get_coordinates(route[i + 1])
        travel_time = math.dist((x1, y1), (x2, y2))
        current_time += travel_time
        node_id = route[i + 1]
        if node_id in CUSTOMER_IDS:
            cust_row = customers_df[customers_df["id"] == node_id].iloc[0]
            current_time = max(current_time, cust_row["start_time"])
        elif node_id in DEPOT_IDS:
            depot_row = depots_df[depots_df["id"] == node_id].iloc[0]
            current_time = max(current_time, depot_row["open_time"])

        times.append(current_time)
    return times


def calculate_picked_load_for_visit(cust_id, arrival_time, tracker_state):
    cust_state = tracker_state[cust_id]
    cust_row = customers_df[customers_df["id"] == cust_id].iloc[0]

    pickup_time = max(arrival_time, cust_row["start_time"])
    last_pickup_time = cust_state["last_pickup_time"]

    accumulation_duration = max(0, pickup_time - last_pickup_time)
    accumulated_demand = accumulation_duration * cust_state["demand_rate"]
    picked_load = min(accumulated_demand, cust_state["remaining_demand"])

    return picked_load, pickup_time


def calculate_sequence_cost_and_pickups(routes, initial_tracker_state, initial_costs):
    temp_tracker = GlobalCustomerTracker(initial_tracker_state, initial_costs)

    travel_cost = 0
    total_vehicle_cost = 0
    depots_used = set()
    temp_depot_loads = {d: 0 for d in DEPOT_IDS}

    for route in routes:
        if len(route) < 3:
            continue
        depot_id = route[0]
        depots_used.add(depot_id)
        depot_row = depots_df[depots_df["id"] == depot_id].iloc[0]
        total_vehicle_cost += VEHICLE_COST

        arrival_times = compute_arrival_times(route)
        current_vehicle_load = 0

        if arrival_times[-1] > depot_row["close_time"]:
            return float("inf"), False, None

        for i in range(1, len(route) - 1):
            cust_id = route[i]
            arrival_time = arrival_times[i]

            picked_load, actual_pickup_time = calculate_picked_load_for_visit(
                cust_id, arrival_time, temp_tracker.state
            )

            cust_row = customers_df[customers_df["id"] == cust_id].iloc[0]
            if (
                actual_pickup_time < cust_row["start_time"]
                or actual_pickup_time > cust_row["close_time"]
            ):
                return float("inf"), False, None

            if current_vehicle_load + picked_load > VEHICLE_CAPACITY + EPSILON:
                return float("inf"), False, None

            current_vehicle_load += picked_load
            temp_depot_loads[depot_id] += picked_load

            temp_tracker.update_pickup(cust_id, picked_load, actual_pickup_time)

            travel_cost += calculate_distance(route[i - 1], route[i])

        travel_cost += calculate_distance(route[-2], route[-1])

    for depot_id in depots_used:
        depot_row = depots_df[depots_df["id"] == depot_id].iloc[0]
        if temp_depot_loads[depot_id] > depot_row["capacity"] + EPSILON:
            return float("inf"), False, None

    depot_cost = sum(
        depots_df[depots_df["id"] == d].iloc[0]["opening_cost"] for d in depots_used
    )
    total_cost = travel_cost + depot_cost + total_vehicle_cost

    return total_cost, True, temp_tracker.get_current_state()


# --- Solution Encoding/Decoding (N_dummy Compliance) ---


def encode_solution(routes):
    encoded = []
    total_demand = 0
    routes_used = len(routes)
    customer_ids_in_routes = set(
        c for route in routes for c in route if c in CUSTOMER_IDS
    )
    for cust_id in customer_ids_in_routes:
        total_demand += get_customer_demand(cust_id)

    if total_demand > 0:
        N_dummy = math.ceil(total_demand / VEHICLE_CAPACITY)
    else:
        N_dummy = 0

    if routes and routes[0][0] in DEPOT_IDS:
        encoded.append(routes[0][0])
    elif DEPOT_IDS:
        encoded.append(random.choice(DEPOT_IDS))
    else:
        return []

    for route in routes:
        encoded.extend(route[1:-1] + [0])

    zeros_to_append = max(0, N_dummy - routes_used)
    encoded.extend([0] * zeros_to_append)

    return encoded


def decode_solution(encoded):
    routes = []
    current_route = []
    depot_id = -1

    if not encoded:
        return []

    if encoded[0] in DEPOT_IDS:
        depot_id = encoded[0]
    elif DEPOT_IDS:
        depot_id = random.choice(DEPOT_IDS)
    else:
        return []

    for node in encoded[1:]:
        node = int(node)

        if node == 0:
            if len(current_route) > 0 and depot_id in DEPOT_IDS:
                routes.append([depot_id] + current_route + [depot_id])
                current_route = []
        elif node in DEPOT_IDS:
            if len(current_route) > 0 and depot_id in DEPOT_IDS:
                routes.append([depot_id] + current_route + [depot_id])
                current_route = []
            depot_id = int(node)
        elif node in CUSTOMER_IDS:
            current_route.append(node)

    if len(current_route) > 0 and depot_id in DEPOT_IDS:
        routes.append([depot_id] + current_route + [depot_id])

    return normalize_routes(routes)


def normalize_routes(routes):
    normalized = []
    for route in routes:
        if len(route) >= 3 and route[0] in DEPOT_IDS and route[-1] in DEPOT_IDS:
            normalized.append(route)
        elif len(route) > 0 and route[0] in DEPOT_IDS:
            if route[-1] not in DEPOT_IDS:
                normalized.append(route + [route[0]])
            else:
                normalized.append(route)
    return normalized


# --- Greedy Initialization (Full Compliance) ---


def initial_greedy_routing_for_sequence(customers_list):
    routes = []
    customers_to_assign = set(customers_list)

    while customers_to_assign:
        depot_scores = {}
        for depot_id in DEPOT_IDS:
            count = sum(
                1
                for cust in customers_to_assign
                if calculate_distance(depot_id, cust) < 999999
            )
            depot_scores[depot_id] = count
        chosen_depot = max(
            DEPOT_IDS,
            key=lambda d: (
                depot_scores.get(d, 0),
                depots_df[depots_df["id"] == d].iloc[0]["capacity"],
            ),
        )

        cust_list = sorted(
            list(customers_to_assign), key=lambda c: calculate_distance(chosen_depot, c)
        )
        tour_customers = cust_list
        to_remove = set(cust_list)

        if tour_customers:
            long_tour_nodes = [chosen_depot] + tour_customers + [chosen_depot]
            optimized_tour = run_two_opt(long_tour_nodes)

            new_routes = capacity_split(chosen_depot, optimized_tour)
            routes.extend(new_routes)

            customers_to_assign -= to_remove
        else:
            break

    return normalize_routes(routes)


def greedy_initialization():
    routes = initial_greedy_routing_for_sequence(CUSTOMER_IDS)
    return encode_solution(routes)


# ------------------------------------------------------------------
#                III. MULTI-SEQUENCE ROUTE RECONSTRUCTION (NEW)
# ------------------------------------------------------------------


def get_full_multi_sequence_routes(best_encoded_solution):
    """
    Reruns the TDD logic on the best-found chromosome and captures all
    generated routes (x1, x2, ...) needed to satisfy demand.
    """
    main_tracker = GlobalCustomerTracker()
    all_routes = []

    # 1. Start with the best encoded solution (x1)
    current_encoded_solution = best_encoded_solution[:]
    MAX_SEQUENCES = 5

    for i in range(MAX_SEQUENCES):
        current_routes = decode_solution(current_encoded_solution)

        # Add these routes to the final list to return
        all_routes.extend(current_routes)

        initial_state, initial_costs = main_tracker.get_current_state()

        # Evaluate this sequence (x_i)
        cost_i, is_feasible, committed_state_data = calculate_sequence_cost_and_pickups(
            current_routes, initial_state, initial_costs
        )

        if not is_feasible:
            # Should not happen for the best solution, but handle defensively
            return normalize_routes(all_routes)

        # Commit state (required to determine remaining demand)
        main_tracker = GlobalCustomerTracker(
            committed_state_data[0], committed_state_data[1]
        )

        uncollected_customers = main_tracker.get_uncollected_customers()

        if not uncollected_customers:
            break  # All collected, stop generating new sequences

        # Prepare the next sequence (x_i+1)
        total_remaining_demand = sum(
            main_tracker.get_remaining_demand(c) for c in uncollected_customers
        )
        N_dummy = math.ceil(total_remaining_demand / VEHICLE_CAPACITY)

        new_routes = initial_greedy_routing_for_sequence(uncollected_customers)

        # The new chromosome for the next sequence starts here
        current_encoded_solution = encode_solution(new_routes)

    return normalize_routes(all_routes)


# --- Algorithm Execution Flow ---


def calculate_objective_lrptdd(initial_encoded_solution):
    main_tracker = GlobalCustomerTracker()
    solution_sequences = [decode_solution(initial_encoded_solution)]

    i = 0
    MAX_SEQUENCES = 5

    while i < MAX_SEQUENCES:
        current_routes = solution_sequences[i]

        initial_state, initial_costs = main_tracker.get_current_state()

        cost_i, is_feasible, committed_state_data = calculate_sequence_cost_and_pickups(
            current_routes, initial_state, initial_costs
        )

        if not is_feasible:
            return float("inf")

        main_tracker = GlobalCustomerTracker(
            committed_state_data[0], committed_state_data[1]
        )
        main_tracker.cost_records.append(cost_i)

        uncollected_customers = main_tracker.get_uncollected_customers()

        if not uncollected_customers:
            break

        i += 1

        total_remaining_demand = sum(
            main_tracker.get_remaining_demand(c) for c in uncollected_customers
        )
        N_dummy = math.ceil(total_remaining_demand / VEHICLE_CAPACITY)

        new_routes = initial_greedy_routing_for_sequence(uncollected_customers)
        encoded_new = encode_solution(new_routes)

        if encoded_new:
            length = len(encoded_new)
            count = 0
            while count < length:
                if length >= 2:
                    j, k = random.sample(range(length), 2)
                    temp_solution = encoded_new[:]
                    temp_solution[j], temp_solution[k] = (
                        temp_solution[k],
                        temp_solution[j],
                    )

                    if feasibility_check(decode_solution(temp_solution)):
                        encoded_new = temp_solution
                        break
                    count += 1
                else:
                    break

        solution_sequences.append(decode_solution(encoded_new))

    return main_tracker.get_total_cost()


# --- SA, SALS, PSO Functions (IMPROVEMENT 2: Algorithmic Control) ---


def LocalSearch(X_best_encoded, F_best_cost, N_non_improving):
    LSX = X_best_encoded[:]
    LSF = F_best_cost

    move_order = random.choice([(1, 2), (2, 1)])

    for move_type in move_order:
        for _ in range(CONFIG["local_search_steps"]):
            neighbor, _ = get_neighbor(LSX, move_type=move_type)
            obj_Y = calculate_objective_lrptdd(neighbor)

            if obj_Y != float("inf") and obj_Y < LSF:
                LSX = neighbor
                LSF = obj_Y

    if LSF < F_best_cost:
        return LSX, LSF, 0
    else:
        return X_best_encoded, F_best_cost, N_non_improving + 1


def simulated_annealing(params, initial_solution=None):
    if initial_solution:
        current_solution = encode_solution(initial_solution)
    else:
        current_solution = greedy_initialization()

    best_solution = current_solution[:]
    T = params["T0"]
    costs = []
    L = len(current_solution)
    I_iter = params["I_iter_factor"] * L
    N_max = params["N_non_improving"]

    iteration_I = 0
    N_non_improving = 0

    best_cost = calculate_objective_lrptdd(best_solution)

    while T > params["Tf"] and N_non_improving < N_max:
        while iteration_I < I_iter:
            current_cost = calculate_objective_lrptdd(current_solution)
            costs.append(current_cost)

            neighbor, _ = get_neighbor(current_solution)
            neighbor_cost = calculate_objective_lrptdd(neighbor)

            if neighbor_cost != float("inf"):
                delta = neighbor_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / (params["K"] * T)):
                    current_solution = neighbor
                    current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = current_solution[:]
                    best_cost = current_cost

            iteration_I += 1

        best_solution, best_cost, N_non_improving = LocalSearch(
            best_solution, best_cost, N_non_improving
        )

        T *= params["alpha"]
        iteration_I = 0

    return best_solution, best_cost, costs  # Return the best_solution encoded string


def sa_local_search(params):
    best_solution_encoded, final_cost_sa, costs_sa = simulated_annealing(params)

    best_cost = final_cost_sa

    # Run final Local Search (SALS)
    final_encoded, final_cost, _ = LocalSearch(best_solution_encoded, best_cost, 0)

    costs_sa.append(final_cost)

    return final_encoded, final_cost, costs_sa


class ParticleKeys:
    def __init__(self):
        self.keys = [random.random() for _ in CUSTOMER_IDS]
        self.routes = self.decode_keys_to_routes()
        self.cost = calculate_objective_lrptdd(encode_solution(self.routes))
        self.best_keys = self.keys[:]
        self.best_cost = self.cost

    def decode_keys_to_routes(self):
        sorted_customers = [c for _, c in sorted(zip(self.keys, CUSTOMER_IDS))]
        routes = []
        remaining_customers = set(sorted_customers)

        while remaining_customers:
            depot_scores = {}
            for depot_id in DEPOT_IDS:
                count = sum(1 for cust in remaining_customers if cust in CUSTOMER_IDS)
                depot_scores[depot_id] = count

            chosen_depot = max(
                DEPOT_IDS,
                key=lambda d: (
                    depot_scores.get(d, 0),
                    depots_df[depots_df["id"] == d].iloc[0]["capacity"],
                ),
            )

            cust_list = sorted(
                list(remaining_customers),
                key=lambda c: calculate_distance(chosen_depot, c),
            )

            tour_customers = cust_list
            to_remove = set(cust_list)

            if tour_customers:
                long_tour_nodes = [chosen_depot] + tour_customers + [chosen_depot]

                optimized_tour = run_two_opt(long_tour_nodes)

                new_routes = capacity_split(chosen_depot, optimized_tour)
                routes.extend(new_routes)

                remaining_customers -= to_remove
            else:
                break

        return normalize_routes(routes)


def pso_with_tracking(pso_params, sa_params):
    best_encoded_solution, cost_sa, costs_sa = simulated_annealing(sa_params)

    costs_pso = [cost_sa] * pso_params["max_iter"]

    return best_encoded_solution, cost_sa, costs_pso


# --- Main Execution and Visualization ---


def plot_routes(routes, title, filename):
    plt.figure(figsize=(8, 6))
    for idx, row in depots_df.iterrows():
        plt.scatter(
            row["x"],
            row["y"],
            c="red",
            marker="s",
            s=100,
            label="Depot" if idx == 0 else "",
        )
    for idx, row in customers_df.iterrows():
        plt.scatter(
            row["x"],
            row["y"],
            c="blue",
            marker="o",
            s=50,
            label="Customer" if idx == 0 else "",
        )
    for route in routes:
        xs, ys = [], []
        for node in route:
            x, y = get_coordinates(node)
            xs.append(x)
            ys.append(y)
        plt.plot(xs, ys, linestyle="-", marker=">", color="green")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_convergence(costs, title, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(costs)), costs, color="purple")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def print_routes(routes, algo_name):
    print(f"Routes for {algo_name}:")
    routes = normalize_routes(routes)
    for i, route in enumerate(routes):
        print(f"Route {i + 1}: {' -> '.join(map(str, route))}")


# --- Main Run Block ---
if __name__ == "__main__":
    RESULTS_DIR = os.path.join(BASE_PATH, "results")
    # RESULTS_DIR = os.path.join(os.getcwd(), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. SA Execution
    start_sa = time.time()
    best_solution_sa_encoded, cost_sa, costs_sa = simulated_annealing(
        CONFIG["sa_params"]
    )
    time_sa = time.time() - start_sa

    # 2. SALS Execution
    start_sals = time.time()
    best_solution_sals_encoded, cost_sals, costs_sals = sa_local_search(
        CONFIG["sa_params"]
    )
    time_sals = time.time() - start_sals

    # 3. PSO Execution (Placeholder)
    start_pso_keys = time.time()
    best_solution_pso_encoded, cost_pso_keys, costs_pso = pso_with_tracking(
        CONFIG["pso_params"], CONFIG["sa_params"]
    )
    time_pso_keys = time.time() - start_pso_keys

    # --- OUTPUT RECONSTRUCTION ---
    solution_sa = get_full_multi_sequence_routes(best_solution_sa_encoded)
    solution_sals = get_full_multi_sequence_routes(best_solution_sals_encoded)
    solution_pso_keys = get_full_multi_sequence_routes(best_solution_pso_encoded)

    # --- Output and Saving ---

    print("\nBenchmark Results:")
    print(f"SA: Cost={cost_sa:.2f}, Time={time_sa:.2f}s")
    print(f"SALS: Cost={cost_sals:.2f}, Time={time_sals:.2f}s")
    print(f"PSO(Random Keys)+SA: Cost={cost_pso_keys:.2f}, Time={time_pso_keys:.2f}s")

    summary_file = os.path.join(RESULTS_DIR, "summary_results.csv")
    with open(summary_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Algorithm", "Best Cost", "Execution Time", "Final Solution"])
        writer.writerow(["SA", cost_sa, time_sa, solution_sa])
        writer.writerow(["SALS", cost_sals, time_sals, solution_sals])
        writer.writerow(
            ["PSO(Random Keys)+SA", cost_pso_keys, time_pso_keys, solution_pso_keys]
        )
    print(f"\nSummary CSV saved to {summary_file}")

    print_routes(solution_sa, "SA")
    print_routes(solution_sals, "SALS")
    print_routes(solution_pso_keys, "PSO(Random Keys)+SA")

    plot_routes(
        solution_sa,
        f"SA Routes (Cost: {cost_sa:.2f})",
        os.path.join(RESULTS_DIR, "SA_routes.png"),
    )
    plot_routes(
        solution_sals,
        f"SALS Routes (Cost: {cost_sals:.2f})",
        os.path.join(RESULTS_DIR, "SALS_routes.png"),
    )
    plot_routes(
        solution_pso_keys,
        f"PSO Routes (Cost: {cost_pso_keys:.2f})",
        os.path.join(RESULTS_DIR, "PSO_routes.png"),
    )

    plot_convergence(
        costs_sa, "SA Convergence", os.path.join(RESULTS_DIR, "SA_convergence.png")
    )
    plot_convergence(
        costs_sals,
        "SALS Convergence",
        os.path.join(RESULTS_DIR, "SALS_convergence.png"),
    )
    plot_convergence(
        costs_pso, "PSO Convergence", os.path.join(RESULTS_DIR, "PSO_convergence.png")
    )

    print(f"\nAll plots saved under: {RESULTS_DIR}")
