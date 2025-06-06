from .abstract_solver import AbstractSolver
import gurobipy as gp
from gurobipy import GRB
import numpy as np

class solver_343420(AbstractSolver):
    def __init__(self, env):
        super().__init__(env)
        self.name = 'OR_project_problem'
    
    def solve(self):
        super().solve()
        
        # Get instance data
        weights = self.env.inst.weights
        service = self.env.inst.service  # [n_depositis x n_supermarkets]
        distances = self.env.inst.distances  # [(n_depositis+1) x (n_depositis+1)]
        
        n_depositis = service.shape[0]
        n_supermarkets = service.shape[1]
        
        model = gp.Model(self.name)
        model.setParam('OutputFlag', 1)  # Enable output
        
        #-- decision variables--

        # X[i] = 1 if deposit i is built, 0 otherwise
        X = model.addVars(n_depositis, vtype=GRB.BINARY, name="deposit")
        
        # Y[i,j] = 1 if vehicle go from location i to location j, 0 otherwise
        # Location 0 is the company, locations 1 to n_depositis are depositis
        Y = model.addVars(n_depositis + 1, n_depositis + 1, vtype=GRB.BINARY, name="route")
        
        # Z[s] = 1 if supermarket s is NOT served by at least one deposit, 0 otherwise
        Z = model.addVars(n_supermarkets, vtype=GRB.BINARY, name="served")
        
        # Subtour elimination variables (MTZ formulation)
        U = model.addVars(n_depositis + 1, vtype=GRB.CONTINUOUS, name="order", lb=0)
        
        #-- ojective function --

        construction_cost = gp.quicksum(X[i] * weights['construction'] for i in range(n_depositis))
        missed_supermarket_cost = gp.quicksum(Z[i] * weights['missed_supermarket'] for i in range(n_supermarkets))
        #y[0,0] if there are no deposits, it means the company is not serving deposit
        travel_cost = gp.quicksum(Y[i,j] * distances[i,j] * weights['travel'] + Y[0,0] for i in range(n_depositis + 1) for j in range(n_depositis + 1))
        
        model.setObjective(construction_cost + missed_supermarket_cost + travel_cost, GRB.MINIMIZE)
        
        #-- constraints --
        
        # supermarket is served if at least one deposit that can serve it is built
        for s in range(n_supermarkets):
            model.addConstr(
                (1- Z[s]) <= gp.quicksum(X[i] * service[i,s] for i in range(n_depositis)),
                f"supermarket_service_{s}"
            )
        
        # only travel to or from a deposit if it built
        for i in range(1, n_depositis + 1):  # deposit locations (1 to n_depositis)
            # Outgoing edges
            model.addConstr(
                gp.quicksum(Y[i,j] for j in range(n_depositis + 1) if j != i) <= X[i-1],
                f"outgoing_deposit_{i}"
            )
            # Incoming edges
            model.addConstr(
                gp.quicksum(Y[j,i] for j in range(n_depositis + 1) if j != i) <= X[i-1],
                f"incoming_deposit_{i}"
            )
        
        # 3. Flow conservation constraints
        # Company (location 0): one outgoing, one incoming edge if any depositis are built
        # total_depositis = gp.quicksum(X[i] for i in range(n_depositis))
        model.addConstr(
            gp.quicksum(Y[0,j] for j in range(1, n_depositis + 1)) == 1,
            "company_outgoing"
        )
        model.addConstr(
            gp.quicksum(Y[j,0] for j in range(1, n_depositis + 1)) == 1,
            "company_incoming"
        )
        
        # entries = exits for each deposit
        for i in range(1, n_depositis + 1):
            model.addConstr(
                gp.quicksum(Y[i,j] for j in range(n_depositis + 1) if j != i) == X[i-1],
                f"flow_out_{i}"
            )
            model.addConstr(
                gp.quicksum(Y[j,i] for j in range(n_depositis + 1) if j != i) == X[i-1],
                f"flow_in_{i}"
            )
        
        #no self-loops
        for i in range(n_depositis + 1):
            model.addConstr(Y[i,i] == 0, f"no_self_loop_{i}")
        
        # 5. Subtour elimination (Miller-Tucker-Zemlin formulation)
        M = n_depositis + 1  # Big M
        
        for i in range(1, n_depositis + 1):
            for j in range(1, n_depositis + 1):
                if i != j:
                    model.addConstr(
                        U[i] - U[j] + M * Y[i,j] <= M - 1,
                        f"mtz_{i}_{j}"
                    )
        
        # Company has order 0
        model.addConstr(U[0] == 0, "company_order")
        
        # deposit order bounds
        for i in range(1, n_depositis + 1):
            model.addConstr(U[i] <= M * X[i-1], f"order_bound_{i}")
        
        # solve 
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            print(f"Optimal solution found with cost: {model.objVal}")
            
            # get the values of X from the solver
            X_sol = []
            for i in range(n_depositis):
                X_sol.append(round(X[i].x))  # round is needed to fix decimals

            X_sol = np.array(X_sol, dtype=int) #array of int

            Y_sol = np.zeros((n_depositis + 1, n_depositis + 1), dtype=int) #array of int

            # fill the Y with the values from the solver
            for i in range(n_depositis + 1):
                for j in range(n_depositis + 1):
                    Y_sol[i][j] = round(Y[i, j].x)

            
            # print solution 
            built_depositis = np.where(X_sol == 1)[0]
            print(f"Built depositis: {built_depositis}")
            
            served_supermarkets = 0
            for s in range(n_supermarkets):
                if any(service[i,s] == 1 and X_sol[i] == 1 for i in range(n_depositis)):
                    served_supermarkets += 1
            
            print(f"Served supermarkets: {served_supermarkets}/{n_supermarkets}")
            print(f"Construction cost: {np.sum(X_sol) * weights['construction']}")
            print(f"Missed supermarket cost: {(n_supermarkets - served_supermarkets) * weights['missed_supermarket']}")
            
            travel_distance = np.sum(Y_sol * distances)
            print(f"Travel distance: {travel_distance}")
            print(f"Travel cost: {travel_distance * weights['travel']}")
            
            return X_sol, Y_sol
            
        else:
            print(f"Optimization failed!!!")
            return np.zeros(n_depositis, dtype=int), np.zeros((n_depositis + 1, n_depositis + 1), dtype=int)