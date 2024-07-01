module StateSpaceEconMTKExt

using ModelBaseEcon, StateSpaceEcon, ModelingToolkit
using StateSpaceEcon.StackedTimeSolver: StackedTimeSolverData
using StateSpaceEcon.SteadyStateSolver: SolverData as SteadyStateSolverData

##############################
# Stacked time system
##############################

"""
    stacked_time_system(m::Model, exog_data::Matrix; fctype = fcgiven)

Convert a `ModelBaseEcon.Model` into a `ModelingToolkit.NonlinearSystem`
that incorporates the stacked time algorithm.

# Inputs
- `m::Model`: Model to convert.
- `exog_data::AbstractArray{Float64,2}`: Data matrix of size `(NT, nvars + nshks)`,
  where `NT` is the number of simulation periods to simulate (plus lags and leads),
  and `nvars` and `nshks` are the number of variables and shocks in the model.
  This data is used to specify the exogenous data, initial conditions, and
  (if applicable) the final conditions.

# Options
- `fctype`: The class of final conditions to use in the simulation.
  The default is [`fcgiven`](@ref).

!!! note
    If `fctype` is [`fclevel`](@ref) or [`fcslope`](@ref),
    `m` will need its steady state solved prior to calling this function.
    See [`sssolve!`](@ref) or [`solve_steady_state!`](@ref).

# Example
This function is used to bring a `ModelBaseEcon.Model` into
the ModelingToolkit/SciML ecosystem.
Here is an example of calling this function
and then converting the returned system into a `NonlinearProblem`
to be solved with one of the solvers from [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl):
```julia
using ModelBaseEcon, StateSpaceEcon, ModelingToolkit, NonlinearSolve
@using_example E3
m = E3.newmodel()
exog_data = rand(102, 6) # Replace with desired data.
s = create_system(m, exog_data)
nf = NonlinearFunction(s)
u0 = zeros(length(unknowns(s)))
prob = NonlinearProblem(nf, u0)
solver = NewtonRaphson() # Replace with desired solver.
sol = solve(prob, solver)
```

See the [ModelingToolkit.jl docs](https://docs.sciml.ai/ModelingToolkit/stable/tutorials/ode_modeling/)
to see what one can do with the solution object `sol`.
"""
function stacked_time_system(m::Model, exog_data::AbstractArray{Float64,2}; fctype = fcgiven)

    nvars = nvariables(m)
    nshks = nshocks(m)

    NT = size(exog_data, 1)
    plan = Plan(m, 1+m.maxlag:NT-m.maxlead)

    data = zeroarray(m, plan)
    data[m.maxlag+1:end-m.maxlead, nvars.+(1:nshks)] = exog_data[m.maxlag+1:end-m.maxlead, nvars.+(1:nshks)] # shks
    data[1:m.maxlag, :] = exog_data[1:m.maxlag, :] # Initial conditions
    if fctype === fcgiven
        data[end-m.maxlead+1:end, :] = exog_data[end-m.maxlead+1:end, :] # Final conditions
    end

    # TODO: Should other variants be accepted, or is `:default` sufficient?
    sd = StackedTimeSolverData(m, plan, fctype, :default)

    return stacked_time_system(sd, data)

end

"""
    stacked_time_system(sd::StackedTimeSolverData, data::Matrix)

Alternative call signature when `sd` is already available (e.g., created manually).
"""
function stacked_time_system(sd::StackedTimeSolverData, data::AbstractArray{Float64,2})

    f = let sd = sd, data = data
        (u, p) -> compute_residuals_stacked_time(u, sd, data)
    end

    # Since we are just creating a MTK system out of the `NonlinearProblem`,
    # the `u0` we specify here is not actually used for solving the system.
    u0 = zeros(count(sd.solve_mask))
    prob = NonlinearProblem(f, u0)
    s = _create_system(prob, sd)

    return s

end

"""
    compute_residuals_stacked_time(u, sd::StackedTimeSolverData, data::Matrix)

Compute the residuals of the stacked time system in `sd`
given variable values `u` and exogenous data `data`.
This function is closed over in [`stacked_time_system`](@ref)
to create a function that can be passed to `NonlinearProblem`.

!!! warning
    Internal function not part of the public interface.
"""
function compute_residuals_stacked_time(u, sd::StackedTimeSolverData, data::AbstractArray{Float64,2})

    # We need to use `zeros` instead of `similar` because `assign_exog_data!`
    # doesn't assign final conditions for the shocks.
    point = zeros(eltype(u), size(data))
    point[sd.solve_mask] = u
    # Note: `assign_exog_data!` assigns both exogenous data (including initial conditions) *and* final conditions.
    assign_exog_data!(point, data, sd)

    # Emulate `StackedTimeSolver.stackedtime_R!`, computing the residual
    # for each equation at each simulation period.
    resid = map(sd.TT[1:length(sd.II)]) do tt
        # `tt` grabs all the data from `point` for the current simulation period, lags, and leads (e.g., `tt = [1, 2, 3, 4, 5, 6]`).
        # The last value of `tt` contains the indices for just the leads (e.g., `tt = [100, 101, 102]`)
        # (but we don't use it in this loop).
        p = view(point, tt, :)
        med = sd.evaldata
        map(med.alleqns, med.allinds) do eqn, inds
            eqn.eval_resid(p[inds])
        end
    end

    return vcat(resid...)

end

"""
    _create_system(prob::NonlinearProblem, sd::StackedTimeSolverData)

Create a `ModelingToolkit.NonlinearSystem` from the given problem.
The solver data `sd` should be the same as used for creating `prob`.

!!! warning
    Internal function not part of the public interface.
"""
function _create_system(prob::NonlinearProblem, sd::StackedTimeSolverData)

    # Convert `prob` into a ModelingToolkit system.
    @named old_sys = modelingtoolkitize(prob)

    # `modelingtoolkitize` creates a system with automatically generated variable names
    # `x₁`, `x₂`, etc.
    # Rename the variables to make it easier to query the solution to problems
    # created from this system.
    sys = rename_variables(old_sys, sd)

    # We need `conservative = true` to prevent errors in `structural_simplify`.
    s = structural_simplify(complete(sys); conservative = true)

    return s

end

"""
    rename_variables(old_sys::NonlinearSystem, sd::StackedTimeSolverData)

Create a new `NonlinearSystem` by replacing the variable names in `old_sys`
with variable names from the solver data `sd`.
The solver data `sd` should be the same as used for creating `old_sys`.

!!! warning
    Internal function not part of the public interface.
"""
function rename_variables(old_sys::NonlinearSystem, sd::StackedTimeSolverData)

    var_names = get_var_names(sd)
    # Compute the number of equations (`NE`) in the system.
    # Assumption: All variables to solve for have the same number of simulation periods that must be solved.
    # Example: The reshaped `sd.solve_mask` is assumed to have the form
    # ```
    # 0 0 0 0 0 0
    # 1 1 1 0 0 0
    # 1 1 1 0 0 0
    # 1 1 1 0 0 0
    # 0 0 0 0 0 0
    # ```
    # not
    # ```
    # 0 0 0 0 0 0
    # 1 1 1 0 0 0
    # 1 0 1 0 0 0
    # 1 1 1 0 0 0
    # 0 0 0 0 0 0
    # ```
    # or something similar.
    NE = count(sd.solve_mask) ÷ length(var_names)

    # For each model variable, create a ModelingToolkit variable vector
    # with the correct number of time steps.
    # `vars` will end up being, e.g., `[pinf[1:97], rate[1:97], ygap[1:97]]`.
    vars = map(var_names) do var_name
        ModelingToolkit.@variables $(var_name)[1:NE]
    end |> x -> reduce(vcat, x)
    # Collect all the individual variables to enable substituting the variable names
    # that were automatically generated by `modelingtoolkitize`.
    # `vars_enumerated` will end up being, e.g., `[pinf[1], pinf[2], ..., ygap[97]]`.
    vars_enumerated = reduce(vcat, map(collect, vars))
    old_vars = unknowns(old_sys)
    subs = Dict(old_vars .=> vars_enumerated)

    # Take all the equations from `old_sys`, rename the variables,
    # and create a new system with the new variable names.
    eqs = substitute.(ModelingToolkit.equations(old_sys), Ref(subs))
    @named sys = NonlinearSystem(eqs, vars_enumerated, [])

    return sys

end

function get_var_names(sd)

    solve_mask = reshape(sd.solve_mask, :, maximum(last, sd.evaldata.var_to_idx))
    var_cols = vec(mapslices(any, solve_mask; dims = 1)) |> findall
    var_names = map(var_cols) do i
        findfirst(p -> last(p) == i, sd.evaldata.var_to_idx)
    end

    return var_names

end

##############################
# Steady state system
##############################

function solve_steady_state!(m)

    ss = solve_steady_state(m).u
    m.sstate.values[.!m.sstate.mask] = ss
    m.sstate.mask .= true

    return m

end

solve_steady_state(m) = solve_steady_state(SteadyStateSolverData(m))

function solve_steady_state(sd::SteadyStateSolverData)

    f = let sd = sd
        (u, p) -> compute_residuals_steady_state(u, sd)
    end

    u0 = zeros(count(sd.solve_var))
    prob = NonlinearProblem(f, u0)
    # TODO: Rename variables?
    @named sys = modelingtoolkitize(prob)
    # TODO: `structural_simplify` fails on model E7A.
#    s = structural_simplify(complete(sys); conservative = true)
    s = complete(sys)
    nf = NonlinearFunction(s)
    prob = NonlinearProblem(nf, zeros(length(unknowns(s))))
    sol = solve(prob, NewtonRaphson())

    return sol

end

# `ModelBaseEcon.__to_dyn_pt` doesn't work with automatic differentiation,
# so copy it here to allow passing in an apropriately typed `buffer`.
# TODO: Do we want to make a PR into ModelBaseEcon.jl to add this method?
function __to_dyn_pt!(buffer, pt, s)
    # This function applies the transformation from steady
    # state equation unknowns to dynamic equation unknowns
    for (i, jt) in enumerate(s.JT)
        if length(jt.ssinds) == 1
            pti = pt[jt.ssinds[1]]
        else
            pti = pt[jt.ssinds[1]] + ModelBaseEcon.__lag(jt, s) * pt[jt.ssinds[2]]
        end
        buffer[i] += pti
    end
    return buffer
end

function compute_residuals_steady_state(u, sd)

    point = zeros(eltype(u), length(sd.solve_var))
    point[sd.solve_var] = u

    resid = map(sd.alleqns) do eqn
        if hasproperty(eqn.eval_resid, :s)
            # `eqn.eval_resid` closes over `s::SSEqData`.
            # There's no function to create just `s`, so just grab it from the closure.
            s = eqn.eval_resid.s
            buffer = zeros(eltype(u), length(s.JT))
            s.eqn.eval_resid(__to_dyn_pt!(buffer, point[eqn.vinds], s))
        else
            # `eqn.eval_resid` is an `EquationEvaluator` (not a closure)
            # that computes the residual for an equation defined by a steady state constraint.
            eqn.eval_resid(point[eqn.vinds])
        end
    end

    return resid

end

end
