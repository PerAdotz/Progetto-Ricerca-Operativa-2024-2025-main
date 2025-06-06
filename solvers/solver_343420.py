from .abstract_solver import AbstractSolver
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from collections import defaultdict
from itertools import combinations

def shortest_subtour(edges):
    """Given a list of edges, return the shortest subtour (as a list of nodes)
    found by following those edges. It is assumed there is exactly one 'in'
    edge and one 'out' edge for every node represented in the edge list."""
    # Create a mapping from each node to its neighbours
    node_neighbors = defaultdict(list)
    for i, j in edges:
        node_neighbors[i].append(j)
    
    # Follow edges to find cycles. Each time a new cycle is found, keep track
    # of the shortest cycle found so far and restart from an unvisited node.
    unvisited = set(node_neighbors)
    shortest = None
    while unvisited:
        cycle = []
        neighbors = list(unvisited)
        while neighbors:
            current = neighbors.pop()
            cycle.append(current)
            unvisited.remove(current)
            neighbors = [j for j in node_neighbors[current] if j in unvisited]
        if shortest is None or len(cycle) < len(shortest):
            shortest = cycle
    
    return shortest if shortest is not None else []

class TSPCallback:
    """Callback class implementing lazy constraints for the TSP. At MIPSOL
    callbacks, solutions are checked for subtours and subtour elimination
    constraints are added if needed."""
    def __init__(self, nodes, x):
        self.nodes = nodes
        self.x = x
    
    def __call__(self, model, where):
        """Callback entry point: call lazy constraints routine when new
        solutions are found."""
        if where == GRB.Callback.MIPSOL:
            try:
                self.eliminate_subtours(model)
            except Exception as e:
                print(f"Exception occurred in MIPSOL callback: {e}")
                model.terminate()
    
    def eliminate_subtours(self, model):
        """Extract the current solution, check for subtours, and formulate lazy
        constraints to cut off the current solution if subtours are found.
        Assumes we are at MIPSOL."""
        values = model.cbGetSolution(self.x)
        edges = [(i, j) for (i, j), v in values.items() if v > 0.5]
        
        if not edges:
            return
            
        tour = shortest_subtour(edges)
        if len(tour) < len(self.nodes):
            # add subtour elimination constraint for every pair of cities in tour
            model.cbLazy(gp.quicksum(self.x[i, j] for i, j in combinations(tour, 2)) <= len(tour) - 1)

class solver_343420(AbstractSolver):
    def __init__(self, env):
        super().__init__(env)
        self.name = 'OR_project_problem'
    
    def solve(self):
        super().solve()
        
        # Get instance data
        weights = self.env.inst.weights
        service = self.env.inst.service  # [n_deposits x n_supermarkets]
        distances = self.env.inst.distances  # [(n_deposits+1) x (n_deposits+1)]
        
        n_deposits = service.shape[0]
        n_supermarkets = service.shape[1]
        
        model = gp.Model(self.name)
        model.setParam('OutputFlag', 1)  # Enable output
        
        #-- decision variables--

        # X[i] = 1 if deposit i is built, 0 otherwise
        X = model.addVars(n_deposits, vtype=GRB.BINARY, name="deposit")
        
        # Y[i,j] = 1 if vehicle go from location i to location j, 0 otherwise
        # location 0 is the company, locations 1 to n_deposits are deposits
        Y = model.addVars(n_deposits + 1, n_deposits + 1, vtype=GRB.BINARY, name="route")

        # Z[s] = 1 if supermarket s is NOT served by at least one deposit, 0 otherwise
        Z = model.addVars(n_supermarkets, vtype=GRB.BINARY, name="missed")
        
        #-- objective function --

        construction_cost = gp.quicksum(X[i] * weights['construction'] for i in range(n_deposits))
        missed_supermarket_cost = gp.quicksum(Z[i] * weights['missed_supermarket'] for i in range(n_supermarkets))
        #y[0,0] if there are no deposits, it means the company is not serving deposit
        travel_cost = gp.quicksum(Y[i,j] * distances[i,j] * weights['travel'] for i in range(n_deposits + 1) for j in range(n_deposits + 1))

        
        model.setObjective(construction_cost + missed_supermarket_cost + travel_cost  + Y[0,0] , GRB.MINIMIZE)
        
        #-- constraints --
        
        # supermarket is served if at least one deposit that can serve it is built
        for s in range(n_supermarkets):
            model.addConstr(
                (1 - Z[s]) <= gp.quicksum(X[i] * service[i,s] for i in range(n_deposits)),
                f"supermarket_service_{s}"
            )
        
        # only travel to or from a deposit if it is built
        for i in range(1, n_deposits + 1):  # deposit locations (1 to n_deposits)
            # uutgoing
            model.addConstr( gp.quicksum(Y[i,j] for j in range(n_deposits + 1) if j != i) <= X[i-1],
                f"outgoing_deposit_{i}"
            )
            # incoming
            model.addConstr(gp.quicksum(Y[j,i] for j in range(n_deposits + 1) if j != i) <= X[i-1],
                f"incoming_deposit_{i}"
            )

        # flow conservation constraints
        # company (location 0): one outgoing, one incoming connection if any depositis are built
        #start and return to the company
        model.addConstr(
            gp.quicksum(Y[0,j] for j in range(1, n_deposits + 1)) == 1,
            "company_outgoing"
        )
        model.addConstr(
            gp.quicksum(Y[j,0] for j in range(1, n_deposits + 1)) == 1,
            "company_incoming"
        )
        
        # entries = exits for each deposit
        for i in range(1, n_deposits + 1):
            model.addConstr(
                gp.quicksum(Y[i,j] for j in range(n_deposits + 1) if j != i) == X[i-1],
                f"flow_out_{i}"
            )
            model.addConstr(
                gp.quicksum(Y[j,i] for j in range(n_deposits + 1) if j != i) == X[i-1],
                f"flow_in_{i}"
            )
        
        #no self-loops
        for i in range(n_deposits + 1):
            model.addConstr(Y[i,i] == 0, f"no_self_loop_{i}")

        # avoid to go from the company to a deposit not built 
        for j in range(1, n_deposits + 1):
            model.addConstr(Y[0,j] <= X[j-1], f"company_to_deposit_{j}")

        
        # subtour elimination with callback
        model.Params.LazyConstraints = 1
        nodes = list(range(n_deposits + 1))  # all possible nodes
        cb = TSPCallback(nodes, Y)
        model.optimize(cb)
        
        if model.status == GRB.OPTIMAL:
            print(f"Optimal solution found with cost: {model.objVal}")
            
            # get the values of X from the solver
            X_sol = []
            for i in range(n_deposits):
                X_sol.append(round(X[i].x))

            X_sol = np.array(X_sol, dtype=int)

            Y_sol = np.zeros((n_deposits + 1, n_deposits + 1), dtype=int)

            # fill the Y with the values from the solver
            for i in range(n_deposits + 1):
                for j in range(n_deposits + 1):
                    if i != j and (i, j) in Y:
                        Y_sol[i][j] = round(Y[i, j].x)
            
            return X_sol, Y_sol
            
        else:
            print(f"Optimization failed!!!")
            return np.zeros(n_deposits, dtype=int), np.zeros((n_deposits + 1, n_deposits + 1), dtype=int)