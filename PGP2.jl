# Import packages
using JuMP, CPLEX

# Data
# Number of generator types
J = 4
# Number of periods in the discretized load duration curve
I = 3

# Budget ($)
b = 220.0
# Minimum capacity (kw)
M = 15.0
# Capital Cost ($/kw)
c = [10.0, 7, 16, 6]
# Operating cost ($/kw hr)
f = [40.0, 45, 32, 55]
# Load duration (hr)
beta = [1.0, 0.6, 0.1]

# Marginal distribution of demand tilde omega
D_values = [
    [0.5, 1.0, 2.5, 3.5, 5.0, 6.5, 7.5, 9.0, 9.5],
    [0.0, 1.5, 2.5, 4.0, 5.5, 6.5, 8.0, 8.5],
    [0.0, 0.5, 1.5, 3.0, 4.5, 5.5, 7.0, 7.5]
]
D_probs = [
    [0.00005, 0.00125, 0.0215, 0.2857, 0.3830, 0.2857, 0.0215, 0.00125, 0.00005],
    [0.0013, 0.0215, 0.2857, 0.3830, 0.2857, 0.0215, 0.00125, 0.00005],
    [0.0013, 0.0215, 0.2857, 0.3830, 0.2857, 0.0215, 0.00125, 0.00005]
]

# Generate the joint distribution from marginal
# Note the usage of splat operator ... in Iterators.product
# Iterators.product takes cartesian product of arguments
p = []
omega = []
for (prob, obs) in zip(
    Iterators.product(D_probs...),
    Iterators.product(D_values...)
)
    push!(p, prob[1] * prob[2] * prob[3])
    push!(omega, obs)
end

# # Alternative
# for i=1:9, j=1:8, k=1:8
#     push!(p, D_probs[1][i] * D_probs[2][j] * D_probs[3][k])
#     push!(omega, (D_values[1][i], D_values[2][j], D_values[3][k]))
# end

# Total number of scenarios
S = length(p)

@show p[1:15]
@show omega[1:15]

# Build the All-in-one LP
model = Model(CPLEX.Optimizer)

# Declare variables and BOUND constraints
@variable(model, x[1:J] >= 0)
@variable(model, y[1:S, 1:I, 1:J] >= 0)

# Specify the objective
@objective(model, Min,
    sum(c[j]*x[j] for j = 1:J)
    + sum(p[s] * sum(f[j] * beta[i] * y[s,i,j] for i in 1:I for j in 1:J) for s in 1:S)
)

# First stage constraints
@constraint(model, c1, sum(c[j] * x[j] for j = 1:J) <= b)
@constraint(model, c2, sum(x[j] for j = 1:J) >= M)

# Second stage constraints. Note the index s as the scenario index
@constraint(model, c3[s=1:S, j=1:J], -x[j] + sum(y[s,i,j] for i = 1:I) <= 0)
@constraint(model, c4[s=1:S, i=1:I], sum(y[s,i,j] for j = 1:J) == omega[s][i])

# Call the optimizer to solve the model
optimize!(model)

# It is usually good to check if the solver actually solved the model
@assert(termination_status(model) == OPTIMAL)

# Querying the objective value
@show objective_value(model)

# Query the decisions. The dot after the function "broadcast" the function
# onto the vector x.
# @show (value(x[1]), value(x[2]), value(x[3]), value(x[4]))
@show value.(x)
