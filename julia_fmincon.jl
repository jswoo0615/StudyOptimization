using JuMP
using Ipopt # 솔버 로딩
using LinearAlgebra # A*x 계산 등에 사용될 수 있음
using MathOptInterface # 최적화 종료 상태 확인을 위해 필요

"""
    fmincon_jump(fun, x0; A, b, Aeq, beq, lb, ub, nonlcon, tol, maxeval, solver_attributes)

MATLAB의 fmincon 함수와 유사하게 비선형 목적 함수, 선형/비선형 제약 조건, 변수 경계 조건을 갖는
최적화 문제를 JuMP를 사용하여 해결하는 함수입니다.

# Arguments
- `fun::Function`: 최소화할 목적 함수 fun(x). x는 Vector{Float64}를 입력받고 Float64를 반환해야 합니다.
- `x0::Vector{Float64}`: 최적화 변수의 초기 추측값. 변수의 차원을 결정하는 데 사용됩니다.

# Keywords
- `A::Union{Matrix{Float64}, Nothing}=nothing`: 선형 부등식 제약 조건 A*x .<= b 의 행렬 A.
- `b::Union{Vector{Float64}, Nothing}=nothing`: 선형 부등식 제약 조건 A*x .<= b 의 벡터 b.
- `Aeq::Union{Matrix{Float64}, Nothing}=nothing`: 선형 등식 제약 조건 Aeq*x .== beq 의 행렬 Aeq.
- `beq::Union{Vector{Float64}, Nothing}=nothing`: 선형 등식 제약 조건 Aeq*x .== beq 의 벡터 beq.
- `lb::Union{Vector{Float64}, Nothing}=nothing`: 변수의 하한 경계 lb .<= x.
- `ub::Union{Vector{Float64}, Nothing}=nothing`: 변수의 상한 경계 x .<= ub.
- `nonlcon::Union{Function, Nothing}=nothing`: 비선형 부등식 제약 조건 nonlcon(x) .<= 0 함수.
  nonlcon(x)는 x(Vector{Float64})를 입력받아 Float64를 반환해야 합니다.
  주의: 현재 JuMP 등록 방식에 따라 nonlcon이 스칼라 값을 반환하는 경우만 `@NLconstraint(..., <= 0)` 형태로 직접 지원 가능합니다.
  여러 비선형 부등식 제약 조건이나 비선형 등식 제약 조건이 있다면, 해당 로직을 추가해야 합니다.
- `tol::Float64=1e-6`: 최적화 해의 상대 오차 허용 한계. (솔버 속성으로 매핑됨)
- `maxeval::Int=10000`: 최대 평가 횟수 또는 반복 횟수. (솔버 속성으로 매핑됨)
- `solver_attributes::Dict{String, Any}=Dict{String, Any}()`: 솔버에 직접 전달할 추가 속성 딕셔너리.

# Returns
- `minx::Vector{Float64}`: 최적화된 변수 값.
- `minf::Float64`: 최적화된 목적 함수 값.
- `status::JuMP.TerminationStatusCode`: 최적화 종료 상태 코드.
"""
function fmincon_jump(fun::Function,
    x0::Vector{Float64};
    A::Union{Matrix{Float64}, Nothing}=nothing,
    b::Union{Vector{Float64}, Nothing}=nothing,
    Aeq::Union{Matrix{Float64}, Nothing}=nothing,
    beq::Union{Vector{Float64}, Nothing}=nothing,
    lb::Union{Vector{Float64}, Nothing}=nothing,
    ub::Union{Vector{Float64}, Nothing}=nothing,
    nonlcon::Union{Function, Nothing}=nothing, # 스칼라 반환 비선형 부등식 제약
    tol::Float64=1e-6,
    maxeval::Int=10000,
    solver_attributes::Dict{String, Any}=Dict{String, Any}())

    model = Model(Ipopt.Optimizer)

    set_attribute(model, "tol", tol)
    set_attribute(model, "max_iter", maxeval)

    for (key, val) in solver_attributes
        set_attribute(model, key, val)
    end

    n = length(x0) # 변수 개수

    # 1. 최적화 변수 선언 및 초기값, 경계 설정
    if lb !== nothing && ub !== nothing
        @variable(model, x[i=1:n], lower_bound = lb[i], upper_bound = ub[i], start = x0[i])
    elseif lb !== nothing
        @variable(model, x[i=1:n], lower_bound = lb[i], start = x0[i])
    elseif ub !== nothing
        @variable(model, x[i=1:n], upper_bound = ub[i], start = x0[i])
    else
        @variable(model, x[i=1:n], start = x0[i])
    end

    # 2. 목적 함수 설정 (@NLobjective)
    # fun 함수(Vector{Float64} -> Float64)를 JuMP에서 사용하기 위해 등록합니다.
    # 등록 함수는 n개의 스칼라 인자를 받도록 래핑합니다.
    obj_wrapper = (args...) -> fun(collect(args))

    # 고유한 심볼 이름으로 함수 등록
    # register(model, 사용할_심볼_이름, 인자_개수, 실제_함수_구현; autodiff=true)
    JuMP.register(model, :fmincon_obj_wrapper, n, obj_wrapper; autodiff=true)

    # 등록된 함수를 @NLobjective에서 사용합니다.
    # x[1:n]... 는 x 배열의 요소들을 개별 인자로 펼쳐서 전달합니다.
    @NLobjective(model, Min, fmincon_obj_wrapper(x[1:n]...))

    # 3. 선형 제약 조건 (@constraint)
    if A !== nothing && b !== nothing
        @constraint(model, lin_ineq_constr, A * x .<= b)
    end

    if Aeq !== nothing && beq !== nothing
        @constraint(model, lin_eq_constr, Aeq * x .== beq)
    end

    # 4. 비선형 제약 조건 (@NLconstraint)
    # nonlcon 함수(Vector{Float64} -> Float64)를 JuMP에서 사용하기 위해 등록합니다.
    # 현재는 스칼라 값을 반환하는 하나의 비선형 부등식 제약만 지원합니다.
    if nonlcon !== nothing
        # nonlcon 함수를 래핑하여 n개의 스칼라 인자를 받도록 합니다.
        nonlcon_wrapper = (args...) -> nonlcon(collect(args))

        # 고유한 심볼 이름으로 함수 등록
        JuMP.register(model, :fmincon_nonlcon_wrapper, n, nonlcon_wrapper; autodiff=true)

        # 등록된 함수를 @NLconstraint에서 사용합니다.
        # 제약 조건은 nonlcon(x) <= 0 형태가 됩니다.
        @NLconstraint(model, nonlin_ineq_constr, fmincon_nonlcon_wrapper(x[1:n]...) <= 0)
    end

    # 5. 최적화 실행
    optimize!(model)

    # 6. 결과 추출
    status = termination_status(model)

    # 최적화 상태가 OPTIMAL 또는 LOCALLY_SOLVED 일 경우 값 추출
    # 이전 논의를 통해 확인된 성공 종료 상태를 사용합니다.
    if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
        minf = objective_value(model)
        minx = value.(x)
    else
        # 성공적으로 해를 찾지 못한 경우 NaN 반환
        minf = NaN
        # n = length(x0) # 변수 개수는 이미 위에 n으로 정의됨
        minx = fill(NaN, n)
        println("Optimization failed. Status: $(status)")
    end

    # 7. 결과 반환
    return minx, minf, status
end