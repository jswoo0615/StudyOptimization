using JuMP
using GLPK                  # LP 솔버, 원한다면 다른 솔버 (HiGHS.Optimizer 등)로 변경 가능합니다.
using LinearAlgebra         # dot 함수 사용
using MathOptInterface      # MOI.OptimizationSense 등을 명시적으로 사용하기 위해 추가

"""
f : 목적 함수 계수 벡터 (minimize f'x)
A, b : Ax <= b 형태의 부등식 제약 조건
Aeq, beq : Aeq * x = beq 형태의 등식 제약 조건
lb, ub : lb <= x <= ub 형태의 변수 경계 제약 조건
optimizer : 사용할 JuMP 옵티마이저 타입 (기본값 : GLPK.Optimizer)
sense : 최적화 방향 (MOI.MIN_SENSE 또는 MOI.MAX_SENSE)
"""

function julia_linprog(f::AbstractVector{Float64};
                       A::Union{AbstractMatrix{Float64}, Nothing}=nothing,
                       b::Union{AbstractVector{Float64}, Nothing}=nothing,
                       Aeq::Union{AbstractMatrix{Float64}, Nothing}=nothing,
                       beq::Union{AbstractVector{Float64}, Nothing}=nothing,
                       lb::Union{AbstractVector{Float64}, Nothing}=nothing,
                       ub::Union{AbstractVector{Float64}, Nothing}=nothing,
                       optimizer=GLPK.Optimizer,
                       sense::MOI.OptimizationSense=MOI.MIN_SENSE,
                       solver_attributes::Dict{String, Any}=Dict{String, Any}())    # 추가 : 솔버 옵션을 담을 딕셔너리 인자
    
    n = length(f)       # 변수의 갯수는 목적 함수 계수 벡터의 길이와 같습니다.

    # 모델 생성 및 솔버 연결
    model = Model(GLPK.Optimizer)

    # ---------- 추가 : 솔버 옵션 설정 ----------
    # solver_attributes 딕셔너리에 담긴 옵션들을 솔버에 전달합니다.
    for (name, value) in solver_attributes
        set_optimizer_attribute(model, name, value)
    end

    # 변수 정의
    # 변수 선언 시 바로 lb, ub 제약을 설정하면 편리합니다.
    @variable(model, x[1:n])

    # 변수 경계 제약 설정 (lb <= x <= ub)
    if lb !== nothing
        @constraint(model, con_lb[i=1:n], x[i] >= lb[i])
        # 또는 set_lower_bound.(x, lb) 사용 가능
    end
    if ub !== nothing
        @constraint(model, con_ub[i=1:n], x[i] <= ub[i])
        # 또는 set_upper_bound.(x, ub) 사용 가능
    end

    # 목적 함수 정의 (sense에 따라 최소화 또는 최대화)
    @objective(model, sense, dot(f, x))     # dot(f, x)는 sum(f[i]*x[i])와 같습니다 (선형 목적 함수).

    # 부등식 제약 조건 설정 (Ax <= b)
    if A !== nothing && b !== nothing
        if size(A, 2) != n
            error("Dimension mismatch : size(A, 2) ($(size(A, 2))) does not match number of variables($n)")
        end
        if size(A, 1) !== length(b)
            error("Dimension mismatch : size(A, 1) ($(size(A, 1))) does not match length(b) ($(length(b)))")
        end
        # 행렬 A의 각 행에 대해 제약을 추가합니다.
        @constraint(model, con_ineq[i=1:size(A, 1)], dot(A[i, :], x) <= b[i])
        # 또는 @constraint(model, con_ineq, Ax .<= b)와 같이 벡터화된 형태로도 가능합니다. (JumP가 자동으로 변환)
    end

    # 등식 제약 조건 설정 (Aeq * x = beq)
    if Aeq !== nothing && b !== nothing
        if size(Aeq, 2) !== n
            error("Dimension mismatch : size(Aeq, 2) ($(size(Aeq, 2))) does not match number of variables ($(n))")
        end
        if size(Aeq, 1) !== length(beq)
            error("Dimension mismatch : size(Aeq, 1) ($(size(Aeq, 1))) does not match length(beq) ($(length(beq)))")
        end
        # 행렬 Aeq의 각 행에 대해 제약을 추가합니다.
        @constraint(model, con_eq[i=1:size(Aeq, 1)], dot(Aeq[i,:], x) == beq[i])
        # 또는 @constraint(model, con_eq, Aeqx .== beq)와 같이 벡터화된 형태로도 가능합니다.
    end

    # 최적화 실행
    optimize!(model)

    # 결과 반환
    # 결과는 튜플이나 커스텀 구조체로 묶어서 반환할 수 있습니다.
    # 여기서는 NamedTuple로 반환합니다.
    if termination_status(model) == MOI.OPTIMAL
        return(
            solution=value.(x),
            objective_value=objective_value(model),
            termination_status=termination_status(model),
            primal_status=primal_status(model)      # 해의 상태 (Optimal, Feasible etc.) 
        )
    else
        return(
            solution=nothing,               # 최적해를 찾지 못함
            objective_value=nothing,
            termination_status=termination_status(model),
            primal_status=primal_status(model)
        )
    end
end