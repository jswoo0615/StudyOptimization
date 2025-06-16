using JuMP
using OSQP
using HiGHS
using LinearAlgebra

"""
quadprog (Quadratic Programming)

- 목적 : 2차 목적 함수 (0.5x^T * H * x + f^T * x)를 선형 제약 조건 및 변수 경계 제약 하에서 최소화

- JuMP.jl + QP 솔버 (OSQP.jl, HiGHS.jl)

Minimize 0.5*x'*H*x + f_linear'*x subject to A*x <= b, Aeq*x == beq, lb <= x <= ub
"""
function julia_quadprog(H::AbstractMatrix{Float64}, f_linear::AbstractVector{Float64};
                        A::Union{AbstractMatrix{Float64}, Nothing}=nothing,
                        b::Union{AbstractVector{Float64}, Nothing}=nothing,
                        Aeq::Union{AbstractMatrix{Float64}, Nothing}=nothing,
                        beq::Union{AbstractVector{Float64}, Nothing}=nothing,
                        lb::Union{AbstractVector{Float64}, Nothing}=nothing,
                        ub::Union{AbstractVector{Float64}, Nothing}=nothing,
                        optimizer=OSQP.Optimizer,
                        solver_attributes::Dict{String, Any}=Dict{String, Any}())

    n = size(H, 1)      # 변수의 갯수는 H 행렬의 크기로 결정 (H는 n x n 행렬)
    if size(H, 2) !== n || length(f_linear) !== n
        error("Dimension mismatch : H must be n x n and f_linear must be of length n")
    end

    model = Model(optimizer)

    for (name, value) in solver_attributes
        set_optimizer_attribute(model, name, value)
    end

    @variable(model, x[1:n])

    if lb !== nothing
        @constraint(model, con_lb[i=1:n], x[i] >= lb)
    end
    if ub !== nothing
        @constraint(model, con_ub[i=1:n], x[i] <= ub)
    end
    if A !== nothing && b !== nothing
        @constraint(model, con_ineq, A * x .<= b)
    end
    if Aeq !== nothing && beq !== nothing
        @constraint(model, con_eq, Aeq * x .== beq)
    end

    # 2차 목적 함수 정의
    @objective(model, Min, 0.5 * x' * H * x + dot(f_linear, x))

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        return (solution = value.(x), objective_value = objective_value(model), termination_status = termination_status(model))
    else
        return (solution = nothing, objective_value = nothing, termination_status = termination_status(model))
    end

end