##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

"""
    sim_nr!(x, sd, maxiter, tol, verbose [, sparse_solver [, linesearch]])

Solve the simulation problem.
  * `x` - the array of data for the simulation. All initial, final and exogenous
    conditions are already in place.
  * `sd::AbstractSolverData` - the solver data constructed for the simulation
    problem.
  * `maxiter` - maximum number of iterations.
  * `tol` - desired accuracy.
  * `verbose` - whether or not to print progress information.
  * `sparse_solver` (optional) - a function called to solve the linear system A
    x = b for x. Defaults to A\\b
  * `linesearch` (optional) - a `true` / `false` controlling whether to run a
    pure Newton-Raphson (`false`) or to include a linesearch at each iteration
    (`true`). Default is `false`.

"""
function sim_nr!(x::AbstractArray{Float64}, sd::StackedTimeSolverData,
    maxiter::Int64, tol::Float64, verbose::Bool,
    sparse_solver::Function=(A, b) -> A \ b, linesearch=false)
    for it = 1:maxiter
        Fx, Jx = global_RJ(x, x, sd)
        nFx = norm(Fx, Inf)
        if nFx < tol
            if verbose
                @info "$it, || Fx || = $(nFx)"
            end
            return true
        end
        Δx = sparse_solver(Jx, Fx)
        λ = 1.0
        if linesearch
            nf = norm(Fx)
            # the Armijo rule: C.T.Kelly, Iterative Methods for Linear and Nonlinear Equations, ch.8.1, p.137
            α = 1e-4
            σ = 0.5
            while λ > 0.00001
                x_buf = copy(x)
                assign_update_step!(x_buf, -λ, Δx, sd)
                nrb2 = try
                    global_R!(Fx, x_buf, x_buf, sd)
                    norm(Fx)
                catch e
                    Inf
                end
                if nrb2 < (1.0 - α * λ) * nf
                    # if verbose && λ < 1.0
                    #     @info "Linesearch success with λ = $λ."
                    # end
                    break
                end
                λ = σ * λ
            end
            if verbose
                if λ <= 0.00001
                    @warn "Linesearch failed."
                elseif λ < 1.0
                    @info "Linesearch success with λ=$λ"
                end
            end
        end
        nΔx = λ * norm(vec(Δx), Inf)
        assign_update_step!(x, -λ, Δx, sd)
        if verbose
            @info "$it, || Fx || = $(nFx), || Δx || = $(nΔx)"
        end
        if nΔx < tol
            return true
        end
    end
    return false
end


export simulate
"""
    simulate(model, plan, data; <options>)

Run a simulation for the given model, simulation plan and exogenous data.

### Arguments
  * `model` - the [`Model`](@ref ModelBaseEcon.Model) instance to simulate.
  * `plan` - the [`Plan`](@ref) for the simulation.
  * `data` - a 2D `Array` containing the exogenous data. This includes the
    initial and final conditions.

### Options as keyword arguments
  * `fctype::`[`FinalCondition`](@ref) - set the desired final condition type
    for the simulation. The default value is [`fcgiven`](@ref). Other possible
    values include [`fclevel`](@ref), [`fcslope`](@ref) and
    [`fcnatural`](@ref).
  * `initial_guess::AbstractMatrix{Float64}` - a 2D `Array` containing the
    initial guess for the solution. This is used to start the Newton-Raphson
    algorithm. The default value is an empty array (`zeros(0,0)`), in which case
    we use the exogenous data for the initial condition. You can use the steady
    state solution using [`steadystatearray`](@ref).
  * `deviation::Bool` - set to `true` if the `data` is given in deviations from
    the steady state. In this case the simulation result is also returned as a
    deviation from the steady state. Default value is `false`.
  * `anticipate::Bool` - set to `false` to instruct the solver that all shocks
    are unanticipated by the agents. Default value is `true`.
  * `verbose::Bool` - control whether or not to print progress information.
    Default value is taken from `model.options`.
  * `tol::Float64` - set the desired accuracy. Default value is taken from
    `model.options`.
  * `maxiter::Int` - algorithm fails if the desired accuracy is not reached
    within this maximum number of iterations. Default value is taken from
    `model.options`.
"""
function simulate end

function simulate(m::Model,
    p_ant::Plan,
    exog_ant::AbstractArray{Float64,2},
    p_unant::Plan=Plan(1U:0U, (;), falses(0, 0)),
    exog_unant::AbstractArray{Float64,2}=zeros(0, 0);
    #= Deviation options =#
    anticipate::Bool=isempty(exog_unant),
    initial_guess::AbstractArray{Float64,2}=zeros(0, 0),
    #= Deviation options =#
    deviation::Bool=false,
    baseline::AbstractArray{Float64,2}=zeros(0, 0),
    deviation_ant=deviation,
    deviation_unant=deviation,
    #= Solver options =#
    verbose::Bool=m.options.verbose,
    tol::Float64=m.options.tol,
    maxiter::Int64=m.options.maxiter,
    fctype=getoption(m, :fctype, fcgiven),
    expectation_horizon::Union{Nothing,Int64}=nothing,
    #= Newton-Raphson options =#
    sparse_solver::Function=(A, b) -> A \ b,
    linesearch=getoption(m, :linesearch, false),
    warn_maxiter=getoption(getoption(m, :warn, Options()), :maxiter, false)
)

    unant_given = !isempty(exog_unant)

    if isempty(p_unant) == unant_given
        error("Invalid `unanticipated` inputs: either plan and data must both be given, or both must be left empty.")
    end

    if anticipate && unant_given
        error("Conflicting arguments: non-empty `exog_unanticipated` with `anticipate=true`.")
    end

    # make sure the model evaluation data is up to date
    refresh_med!(m)

    NT = length(p_ant.range)
    nauxs = length(m.auxvars)
    nvarshks = length(m.varshks)
    logvars = [islog(var) | isneglog(var) for var in m.varshks]

    if size(exog_ant) != (NT, nvarshks)
        error("Incorrect dimensions of exog_data. Expected $((NT, nvarshks)), got $(size(exog_ant)).")
    end
    if !isempty(initial_guess) && size(initial_guess) != (NT, nvarshks)
        error("Incorrect dimensions of initial_guess. Expected $((NT, nvarshks)), got $(size(initial_guess)).")
    end

    if deviation_ant
        exog_ant = copy(exog_ant)
        if isempty(baseline)
            baseline = steadystatearray(m, p_ant)
        end
        if size(baseline) != (NT, nvarshks)
            error("Incorrect dimensions of baseline. Expected $((NT, nvarshks)), got $(size(baseline)).")
        end
        exog_ant[:, logvars] .*= baseline[:, logvars]
        exog_ant[:, .!logvars] .+= baseline[:, .!logvars]
    end
    exog_ant = ModelBaseEcon.update_auxvars(transform(exog_ant, m), m)

    if !isempty(initial_guess)
        x = ModelBaseEcon.update_auxvars(transform(initial_guess, m), m)
    else
        x = copy(exog_ant)
    end

    if anticipate
        gdata = StackedTimeSolverData(m, p_ant, fctype)
        assign_exog_data!(x, exog_ant, gdata)
        if verbose
            @info "Simulating $(p_ant.range[1 + m.maxlag:NT - m.maxlead])" # anticipate gdata.FC
        end
        converged = sim_nr!(x, gdata, maxiter, tol, verbose, sparse_solver, linesearch)
        if warn_maxiter && !converged
            @warn("Newton-Raphson reached maximum number of iterations (`maxiter`).")
        end
    else # unanticipated shocks

        #=== prepare sub-ranges ===#
        init = 1:m.maxlag
        term = NT .+ (1-m.maxlead:0)
        sim = 1+m.maxlag:NT-m.maxlead

        #=== prepare lists of indices according to types of variables ===#
        shkinds = findall(isshock, m.varshks)
        nshks = length(shkinds)

        varinds = findall(!isshock, m.varshks)
        nvars = length(varinds)

        # auxiliary vars are always last
        nvarshks = nvars + nshks
        varshkinds = 1:nvarshks

        nauxs = length(m.auxvars)
        auxinds = nvarshks .+ (1:nauxs)

        nallvars = nvarshks + nauxs
        allvarinds = 1:nallvars

        if unant_given
            #=== check compatibility of unanticipated inputs (data and plan) ===#
            if p_unant.range != p_ant.range
                error("Anticipated and unanticipated ranges don't match.")
            end
            if size(exog_unant) != size(exog_ant)
                error("Anticipated and unanticipated data  don't match.")
            end
            if deviation_unant
                exog_unant[:, logvars] .*= baseline[:, logvars]
                exog_unant[:, .!logvars] .+= baseline[:, .!logvars]
            end
            exog_unant = ModelBaseEcon.update_auxvars(transform(exog_unant, m), m)
        else
            #=== prepare unanticipated data and plan (backward compatibility) ===#
            p_unant = p_ant
            p_ant = Plan(m, p_unant.range[sim])
            exog_unant = copy(exog_ant)
            exog_ant[sim, shkinds] .= 0
            x[sim, shkinds] .= 0
        end

        x[init, allvarinds] .= exog_ant[init, allvarinds]
        t0 = first(sim)
        T = last(sim)
        if expectation_horizon === nothing
            # when expectation_horizon is not given, we simulate each iteration until the end and with the true final condition
            for t in sim
                exog_inds = p_unant[t, Val(:inds)]
                psim = Plan(m, t:T)
                psim.exogenous .= p_ant.exogenous[begin+Int(t - t0):end, :]
                if (t != t0) && (maximum(abs, x[t, exog_inds] .- exog_unant[t, exog_inds]) < tol) #= && (psim[t0, Val(:inds)] == exog_inds) =#
                    continue
                end
                setexog!(psim, t0, exog_inds)
                gdata = StackedTimeSolverData(m, psim, fctype)
                x[t, exog_inds] = exog_unant[t, exog_inds]
                # assign_exog_data!(x[psim.range,:], exog_data[psim.range,:], gdata)
                sim_range = UnitRange{Int}(psim.range)
                xx = view(x, sim_range, :)
                assign_final_condition!(xx, exog_unant[sim_range, :], gdata)
                if verbose
                    @info "Simulating $(p_ant.range[t:T])" # anticipate expectation_horizon gdata.FC
                end
                converged = sim_nr!(xx, gdata, maxiter, tol, verbose, sparse_solver, linesearch)
                if warn_maxiter && !converged
                    @warn("Newton-Raphson reached maximum number of iterations (`maxiter`).")
                end

            end
        else
            # when expectation_horizon is not given,
            # the first and last simulations use the true 
            # simulation range and final condition, while the intermediate 
            # simulations use expectation_horizon steps with fcnatural
            if expectation_horizon == 0
                expectation_horizon = length(sim)
            elseif expectation_horizon < 10 * m.maxlead
                @warn "Expectation horizon may be too short for this model. Consider setting it to at least $(10 * m.maxlead)."
            end
            x = [x; zeros(expectation_horizon, size(x, 2))]
            ninit = length(init)
            nterm = length(term)
            # first simulation
            let t = t0
                # first run is with the full range, the true fctype, 
                # and only the first period is imposed
                exog_inds = p_unant[t, Val(:inds)]
                psim = Plan(m, t:T)
                psim.exogenous .= p_ant.exogenous[begin+Int(t - t0):end, :]
                setexog!(psim, t0, exog_inds)
                sdata = StackedTimeSolverData(m, psim, fctype)
                x[t, exog_inds] = exog_unant[t, exog_inds]
                sim_range = UnitRange{Int}(psim.range)
                xx = view(x, sim_range, :)
                assign_final_condition!(xx, exog_unant[sim_range, :], sdata)
                if verbose
                    @info "Simulating $(p_ant.range[t:T])" # anticipate expectation_horizon sdata.FC
                end
                converged = sim_nr!(xx, sdata, maxiter, tol, verbose, sparse_solver, linesearch)
                if warn_maxiter && !converged
                    @warn("Newton-Raphson reached maximum number of iterations (`maxiter`).")
                end
            end
            # intermediate simulations
            last_t::Int64 = t0
            psim = Plan(m, 0:expectation_horizon-1)
            sdata = StackedTimeSolverData(m, psim, fcnatural)
            for t in sim[2:end]
                exog_inds = p_unant[t, Val(:inds)]
                # we need to run a simulation if a variable is exogenous, or if a shock value is not zero
                # these intermediate simulations are always with fcnatural, 
                #       have length equal to expectation_horizon and 
                #       only the first period is imposed
                if (maximum(abs, x[t, exog_inds] .- exog_unant[t, exog_inds]) < tol) #= && (exog_inds == shkinds) =#
                    continue
                end
                psim1 = copy(psim)
                # the range of psim1 might extend beyond the range of p_ant.
                # we copy from p_ant as far as we have and copy the last line beyond that
                tmp_rng = t:min(t + expectation_horizon - 1, T)
                psim1.exogenous[t0.+(0:length(tmp_rng)-1), :] .= p_ant.exogenous[tmp_rng, :]

                # ===> must leave the psim1 plan empty beyond the end of p_ant
                # becasue we don't have data in exog_and for any exogenized
                # variables.
                # #=
                # for tt = length(tmp_rng)+1:expectation_horizon
                #     psim1.exogenous[t0+tt, :] .= p_ant.exogenous[T, :]
                # end
                # =#

                setexog!(psim1, t0, exog_inds)
                update_plan!(sdata, m, psim1)
                # note that the range always goes from 0 to expectation_horizon-1, 
                # so we need to add t in order to get the correct set of rows of x
                sim_range = t .+ UnitRange{Int}(psim.range)
                xx = view(x, sim_range, :)
                # The initial conditions are already set
                # The exogenous values are already set as well, except for the first period
                # In other words, we only need to impose the first period here
                xx[t0, exog_inds] = exog_unant[t, exog_inds]
                # Update the final conditions (the second argument is not used with fcnatural)
                assign_final_condition!(xx, zeros(0, nallvars), sdata)
                if verbose
                    @info("Simulating $(p_ant.range[t] .+ (0:expectation_horizon - 1))") # anticipate expectation_horizon sdata.FC
                end
                converged = sim_nr!(xx, sdata, maxiter, tol, verbose, sparse_solver, linesearch)
                if warn_maxiter && !converged
                    @warn("Newton-Raphson reached maximum number of iterations (`maxiter`).")
                end
                last_t = t  # keep track of last simulation time
            end
            # last simulation
            if last_t > t0
                # do we need to re-run the last simulation?
                # if it didn't reach T, then yes
                # if the final condition is not fcnatural, then yes
                if (last_t + expectation_horizon != T) || (fctype != fcnatural)
                    psim = Plan(m, min(last_t + 1, T):T)
                    psim.exogenous .= p_ant.exogenous[end.+(1-length(psim.range):0), :]
                    # there are no unanticipated shocks in this simulation
                    sdata = StackedTimeSolverData(m, psim, fctype)
                    # the initial conditions and the exogenous data are already in x
                    # we only need the final conditions
                    sim_range = UnitRange{Int}(psim.range)
                    xx = view(x, sim_range, :)
                    assign_final_condition!(xx, exog_unant[sim_range, :], sdata)
                    if verbose
                        @info "Simulating $(p_ant.range[last_t + 1:T])" # anticipate expectation_horizon sdata.FC
                    end
                    converged = sim_nr!(xx, sdata, maxiter, tol, verbose, sparse_solver, linesearch)
                    if warn_maxiter && !converged
                        @warn("Newton-Raphson reached maximum number of iterations (`maxiter`).")
                    end
                end
            end
            # x = x[begin:end-expectation_horizon, :]
        end
    end

    x = x[axes(exog_ant)...]
    x .= inverse_transform(x, m)
    if deviation
        x[:, logvars] ./= baseline[:, logvars]
        x[:, .!logvars] .-= baseline[:, .!logvars]
    end

    return x
end

# The versions of simulate with Dict/Workspace/SimData

simulate(m::Model, p::Plan, data::AbstractDict; kwargs...) = simulate(m, p, dict2data(data, m, p; copy=true); kwargs...)
simulate(m::Model, p::Plan, data::Workspace; kwargs...) = simulate(m, p, workspace2data(data, m, p; copy=true); kwargs...)

# this is the main interface.
function simulate(m::Model, p::Plan, data::SimData; kwargs...)
    exog = data2array(data, m, p)
    initial_guess = get(kwargs, :initial_guess, nothing)
    if initial_guess isa SimData
        kw = (; initial_guess=data2array(initial_guess, m, p))
    elseif initial_guess isa Workspace
        kw = (; initial_guess=workspace2data(initial_guess, m, p))
    elseif initial_guess isa AbstractDict
        kw = (; initial_guess=dict2array(initial_guess, m, p))
    else
        kw = (;)
    end
    result = copy(data)
    result[p.range, m.varshks] .= simulate(m, p, exog; kwargs..., kw...)
    return result
end



