"""
    MTKExt

A module that is part of StateSpaceEcon.jl.
Declares functions for creating ModelingToolKit systems out of `Model`s.
By default, these functions have no methods;
the relevant methods will be added when ModelingToolKit.jl is loaded.
"""
module MTKExt

gendocs(func_name) = """
    $(func_name)

Function defined in `StateSpaceEconMTKExt`.
Load ModelingToolKit.jl to add relevant methods and additional documentation.
"""

"$(gendocs("stacked_time_system"))"
function stacked_time_system end
export stacked_time_system

function compute_residuals_stacked_time end

function _create_system end

function rename_variables end

function get_var_names end

"$(gendocs("solve_steady_state!"))"
function solve_steady_state! end
export solve_steady_state!

"$(gendocs("steady_state_system"))"
function steady_state_system end
export steady_state_system

function compute_residuals_steady_state end

end

using .MTKExt

export stacked_time_system
export steady_state_system
export solve_steady_state!
