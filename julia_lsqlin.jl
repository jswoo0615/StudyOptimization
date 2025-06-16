using JuMP # JuMP를 사용하여 모델링
using LinearAlgebra # 행렬 연산, 벡터 내적 등을 위해 사용
using OSQP # QP 솔버 예시. Pkg.add("OSQP") 필요.
using MathOptInterface # MOI 상수 (예: MOI.OPTIMAL) 사용을 위해 필요

"""
julia_lsqlin (Constrained Linear Least Squares)

Minimizes sum((C*x - d).^2) subject to linear constraints and bounds.
This is formulated as a Quadratic Programming (QP) problem.

MATLAB의 lsqlin equivalent.

 형태: min ||C*x - d||^2
 제약: A*x <= b
      Aeq*x = beq
      lb <= x <= ub

인자:
- C: Matrix in the objective term ||C*x - d||^2 (AbstractMatrix{Float64}).
- d: Vector in the objective term ||C*x - d||^2 (AbstractVector{Float64}).
- A: Matrix for linear inequality constraints A*x <= b (AbstractMatrix{Float64} or nothing).
- b: Vector for linear inequality constraints A*x <= b (AbstractVector{Float64} or nothing).
- Aeq: Matrix for linear equality constraints Aeq*x = beq (AbstractMatrix{Float64} or nothing).
- beq: Vector for linear equality constraints Aeq*x = beq (AbstractVector{Float64} or nothing).
- lb: Lower bounds for the variables x (AbstractVector{Float64} or nothing).
- ub: Upper bounds for the variables x (AbstractVector{Float64} or nothing).
- optimizer: JuMP-compatible optimizer that supports QP. Default is OSQP.Optimizer.
- solver_attributes: Dictionary of attributes to pass to the optimizer.

반환 값:
Named Tuple 형태의 결과.
(
    solution: 최적화된 변수 벡터 x. value.(x)로 얻음.
    objective_value: 최적화된 변수에서의 목적 함수 값 (sum((C*x - d).^2)).
                     JuMP는 변환된 QP 목적 함수 값을 반환하므로, 원래의 ||C*x - d||^2 값을 계산하여 반환합니다.
    termination_status: JuMP 최적화 종료 상태 (MOI.TerminationStatusCode). JuMP.termination_status(model)로 얻음.
    primal_status: JuMP 프라이멀 상태 (MOI.PrimalStatusCode). 해의 유효성 확인에 유용. JuMP.primal_status(model)로 얻음.
    dual_status: JuMP 듀얼 상태 (MOI.DualStatusCode). JuMP.dual_status(model)로 얻음.
    jump_model: JuMP 모델 객체 (상세 정보 확인용).
)
"""
function julia_lsqlin(C::AbstractMatrix{Float64}, d::AbstractVector{Float64};
                      A::Union{AbstractMatrix{Float64}, Nothing}=nothing,
                      b::Union{AbstractVector{Float64}, Nothing}=nothing,
                      Aeq::Union{AbstractMatrix{Float64}, Nothing}=nothing,
                      beq::Union{AbstractVector{Float64}, Nothing}=nothing,
                      lb::Union{AbstractVector{Float64}, Nothing}=nothing,
                      ub::Union{AbstractVector{Float64}, Nothing}=nothing,
                      optimizer=OSQP.Optimizer,
                      solver_attributes::Dict{String, Any}=Dict{String, Any}())

    # Determine the number of variables (columns of C)
    n = size(C, 2)

    # Input validation
    if size(C, 1) != length(d)
        error("Number of rows in C must match the length of d.")
    end
    if A !== nothing && size(A, 2) != n
        error("Number of columns in A must match the number of columns in C.")
    end
    if A !== nothing && size(A, 1) != length(b)
         error("Number of rows in A must match the length of b.")
     end
    if Aeq !== nothing && size(Aeq, 2) != n
        error("Number of columns in Aeq must match the number of columns in C.")
    end
    if Aeq !== nothing && size(Aeq, 1) != length(beq)
         error("Number of rows in Aeq must match the length of beq.")
     end
    if lb !== nothing && length(lb) != n
        error("Length of lb must match the number of columns in C.")
    end
    if ub !== nothing && length(ub) != n
        error("Length of ub must match the number of columns in C.")
    end
     if lb !== nothing && ub !== nothing && any(lb .> ub)
         error("Lower bounds (lb) must be less than or equal to upper bounds (ub).")
     end


    # Create a JuMP model
    model = Model(optimizer)

    # Set solver attributes
    for (name, value) in solver_attributes
        set_optimizer_attribute(model, name, value)
    end

    # Define variables with bounds if provided
    if lb !== nothing && ub !== nothing
        @variable(model, lb[i] <= x[i=1:n] <= ub[i])
    elseif lb !== nothing
        @variable(model, lb[i] <= x[i=1:n])
    elseif ub !== nothing
        @variable(model, x[i=1:n] <= ub[i])
    else
        @variable(model, x[1:n])
    end

    # Formulate the QP objective: min 0.5*x'*H*x + f'*x
    # where H = 2 * C' * C and f = -2 * C' * d
    H = 2 * C' * C
    f = -2 * C' * d

    @objective(model, Min, 0.5 * x' * H * x + dot(f, x))


    # Add linear inequality constraints A*x <= b
    if A !== nothing && b !== nothing
        @constraint(model, con_ineq, A * x .<= b)
    end

    # Add linear equality constraints Aeq*x = beq
    if Aeq !== nothing && beq !== nothing
        @constraint(model, con_eq, Aeq * x .== beq)
    end

    # Optimize the model
    optimize!(model)

    # Extract results
    # Query status using JuMP.qualification
    term_status = JuMP.termination_status(model) # <-- JuMP. 명시
    primal_status = JuMP.primal_status(model)   # <-- JuMP. 명시
    dual_status = JuMP.dual_status(model)       # <-- JuMP. 명시

    solution = nothing
    objective_val_qp = nothing # Value of the transformed QP objective
    objective_val_lsqlin = nothing # Value of the original ||C*x - d||^2 objective

    # Check primal status for a feasible solution point
    if primal_status == MOI.FEASIBLE_POINT || primal_status == MOI.NEARLY_FEASIBLE_POINT || primal_status == MOI.OPTIMAL_POINT || primal_status == MOI.NEARLY_OPTIMAL_POINT
        # Solution is available
        solution = value.(x)

        # Calculate the original lsqlin objective value: ||C*x - d||^2
        # This can be calculated if a feasible solution is found
        residual_vec = C * solution - d
        objective_val_lsqlin = sum(abs2, residual_vec) # or dot(residual_vec, residual_vec)

         # Also try to get the QP objective value if status is optimal/nearly optimal
         if term_status == MOI.OPTIMAL || term_status == MOI.ALMOST_OPTIMAL
             objective_val_qp = JuMP.objective_value(model) # <-- JuMP. 명시
         end

    else
        # No feasible solution found or optimization failed
        @warn "Optimization did not find a feasible point or terminate optimally. Termination status: $(term_status), Primal status: $(primal_status)"
        # solution, objective values remain nothing
    end

    # Return results as a Named Tuple
    return (
        solution=solution,
        objective_value=objective_val_lsqlin,
        qp_objective_value = objective_val_qp,
        termination_status=term_status,
        primal_status=primal_status,
        dual_status=dual_status,
        jump_model=model # Return the JuMP model for introspection
    )
end