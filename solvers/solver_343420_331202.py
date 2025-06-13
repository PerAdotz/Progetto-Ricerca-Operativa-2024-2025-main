from .abstract_solver import AbstractSolver
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from collections import defaultdict

def shortest_subtour(edges, company_node=0):
    """Given a list of edges, return the shortest subtour (as a list of nodes)
    found by following those edges. It is assumed there is exactly one 'in'
    edge and one 'out' edge for every node represented in the edge list
    Find all subtours that don't include the company node."""
    if not edges:
        return []
    
    # vreate a mapping from each node to its neighbours
    node_neighbors = defaultdict(list)
    for i, j in edges:
        node_neighbors[i].append(j)
    
    # follow edges to find cycles. Each time a new cycle is found, keep track
    # of the shortest cycle found so far and restart from an unvisited node.
    visited = set()
    subtours = []
    
    # find all connected components
    for start_node in node_neighbors:
        if start_node in visited or start_node == company_node:
            continue
    
        component = []
        stack = [start_node]
        
        while stack:
            current = stack.pop()
            if current in visited or current == company_node:
                continue
                
            visited.add(current)
            component.append(current)
            
            # add unvisited neighbors to stack
            for neighbor in node_neighbors[current]:
                if neighbor not in visited and neighbor != company_node:
                    stack.append(neighbor)
        
        # ff we found a component with at least 2 nodes, it's a subtour
        if len(component) >= 2:
            subtours.append(component)
    
    return subtours

class TSPCallback:
    """Callback class implementing lazy constraints for the TSP. At MIPSOL
    callbacks, solutions are checked for subtours and subtour elimination
    constraints are added if needed."""
    
    def __init__(self, y_vars):
        self.y_vars = y_vars
    
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
        values = model.cbGetSolution(self.y_vars)
        edges = [(i, j) for (i, j), v in values.items() if v > 0.5]
        # find subtours (cycles not including company node 0)
        subtours = shortest_subtour(edges, company_node=0)
        
        for subtour in subtours:
            if len(subtour) >= 2:
                # add subtour elimination constraint
                model.cbLazy(
                    gp.quicksum(self.y_vars[i, j] for i in subtour for j in subtour if i != j and (i, j) in self.y_vars) <= len(subtour) - 1
                )

class solver_343420_331202(AbstractSolver):
    def __init__(self, env):
        super().__init__(env)
        self.name = 'DummySolver'
        self.name2 = 'class solver_343420_331202'
    
    def print_solution_edges(self,Y_sol):
        n = Y_sol.shape[0]
        print(f"Solution of the model '{self.name2}' :")
        for i in range(n):
            for j in range(n):
                if Y_sol[i][j] == 1:
                    print(f"{i} -> {j}")

    def solve(self):
        super().solve()
        
        # instance data
        weights = self.env.inst.weights
        service = self.env.inst.service  # [n_deposits x n_supermarkets]
        distances = self.env.inst.distances  # [(n_deposits+1) x (n_deposits+1)]
        
        n_deposits = service.shape[0]
        n_supermarkets = service.shape[1]
        
        model = gp.Model(self.name2)
        
        #decision variables

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
        travel_cost = gp.quicksum(Y[i,j] * distances[i,j] * weights['travel'] for i in range(n_deposits + 1) for j in range(n_deposits + 1))

        #y[0,0] if there are no deposits, it means the company is not serving deposit
        model.setObjective(construction_cost + missed_supermarket_cost + travel_cost + Y[0,0] , GRB.MINIMIZE)
        
        #-- constraints --
        
        # supermarket is served (Z[s] = 0) if at least one deposit that can serve it is built
        for s in range(n_supermarkets):
            model.addConstr(
                (1 - Z[s]) <= gp.quicksum(X[i] * service[i,s] for i in range(n_deposits)),
                f"supermarket_service_{s}"
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

        # only travel to or from a deposit if it is built
        # x[i-1] because array X start from 0 but the deposits start from 1
        # entries = exits for each deposit
        # number of outgoing archs = 1 (if deposit is built -> X[i-1] = 1), 0 (if deposit is not built -> X[i-1] = 0)
        # number of ingoing archs = 1 (if deposit is built -> X[i-1] = 1), 0 (if deposit is not built -> X[i-1] = 0)
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

        
        # subtour elimination with callback similar to guroby website example
        model.Params.LazyConstraints = 1
        cb = TSPCallback(Y)
        model.optimize(cb)
        
        if model.status == GRB.OPTIMAL:
            print(f"\nOptimal solution found with cost: {model.objVal}")
            
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
            
            self.print_solution_edges(Y_sol)
            return X_sol, Y_sol
            
        else:
            print(f"Optimization failed!!!")
            return np.zeros(n_deposits, dtype=int), np.zeros((n_deposits + 1, n_deposits + 1), dtype=int)