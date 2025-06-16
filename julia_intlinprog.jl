using JuMP
using GLPK                  # LP 솔버, 원한다면 다른 솔버 (HiGHS.Optimizer 등)로 변경 가능합니다.
using LinearAlgebra         # dot 함수 사용
using MathOptInterface      # MOI.OptimizationSense 등을 명시적으로 사용하기 위해 추가
"""
intlinprog (Mixed-Integer Linear Programming)

Minimize f'x subject to A*x <= b, Aeq*x = beq, lb <= x <= ub, x[intcon] are Integers
MATLAB의 intlinprog는 Binary 변수도 intcon으로 처리합니다. (0 또는 1은 정수)

여기서는 intcon에 해당하는 변수들에 set_integer()를 적용합니다.
Binary 변수는 0 <= x <= 1 경계와 함께 set_integer()를 사용해야 더욱 명확해집니다.

좀 더 일반적인 형태를 위해 intcon에 해당하는 변수들을 정수 제약으로 설정하는 함수륾 만듭니다.

Binary 변수는 사용자가 lb=0, ub=1, intcon=[변수 인덱스]로 호출하거나, 별도의 binary_var_indices 인자를 받을 수 있습니다.

여기서는 단순화하여 intcon의 변수들에 set_integer를 적용하는 함수를 제시합니다.

f::AbstractVector{Float64}
* Vector{T} : Array{T, 1}의 별칭으로, 특정 타입 (T)의 요소를 담는 1차원 배열 (Vector)의 가장 흔한 구체 타입 (concrete type)입니다.
* AbstractVector{T} : Vector{T}를 포함하여 특정 타입 (T)의 요소를 담는 모든 1차원 배열 타입들의 추상 상위 타입 (abstract supertype)입니다.
                      예를 들어, 배열의 일부를 잘라낸 뷰 (view)나 전치 (transpose)된 벡터 등도 AbstractVector의 하위 타입을 수 있습니다.
* 함수의 인자 타입으로 AbstractVector{Float64}와 같은 추상 타입을 사용하는 것은 함수의 유연성을 높이기 위함입니다.
  이렇게 하면 함수가 표준 Vector{Float64} 뿐만 아니라, Float64 요소를 가진 다른 형태의 1차원 배열 (SubArray{Float64, 1, ...}, LinearAlgebra.Adjoint{Float64, Vector{Float64}} 등)도 인자로 받을 수 있게 됩니다.
* 만약 f::Vector{Float64}라고 했다면, 이 함수는 오직 Vector{Float64} 타입의 인자만 받을 수 있게 되어 활용 범위가 줄어듭니다.

AbstractArray{Int}
* AbstractArray{T}는 특정 타입 (T)의 요소를 담는 모든 차원 (1차원, 2차원, 3차원 등)의 배열 타입들의 추상 상위 타입입니다.
  AbstractVector{T}는 AbstractArray{T}의 하위 타입입니다.
* intcon은 일반적으로 정수 변수의 인덱스 목록이므로 1차원 배열 (Vector)일 가능성이 높습니다. 
  이 경우 AbstractVector{Int}라고 쓰는 것이 더 정확한 타입 힌트일 수 있습니다.

intcon::Union{AbstractVector{Int}, Nothing}=nothing
* intcon : 함수의 인자 이름입니다.
* :: : 타입 어설션 (Type Assertion) 연산자입니다. 이 기호 뒤에는 인자가 가져야 할 타입을 명시합니다.
       여기서는 intcon 인자가 가져야 할 타입이 뒤에 오는 Union{...}이라고 지정하고 있습니다.
* Union{Type1, Type2, ...} : Julia의 합집합 타입 (Union Type)입니다. 
  Union{Type1, Type2}는 변수가 Type1 타입의 값을 가지거나, Type2 타입의 값을 가질 수 있음을 의미합니다.
  여러 타입을 쉼표로 구분하여 나열할 수 있습니다. Union{Type1, Type2, ...} 타입의 변수는 나열된 타입 중 어느 하나에 속하는 값만 담을 수 있습니다.
* AbstractVector{Int} : Union 타입 안에 포함된 첫 번째 타입입니다. 
* Nothing : Union 타입 안에 포함된 두 번째 타입입니다. 
  Nothing은 Julia의 특별한 싱글톤 타입 (Singleton Type)이며, nothing이라는 유일한 값 하나만 가집니다.
  nothing 값은 일반적으로 값이 없음, 정의되지 않음, 결과가 없음 등을 나타낼 때 사용됩니다.
  파이썬의 None과 유사한 역할을 합니다.
* Union{AbstractVector{Int}, Nothing} : 
  이 부분은 intcon 인자가 가질 수 있는 모든 가능한 타입들의 집합을 정의합니다.
  즉, intcon 인자는 AbstractVector{Int} 타입의 값이거나 nothing 값이어야 합니다.
  Float64나 문자열 같은 다른 타입의 값이 오면 타입 오류가 발생합니다.
* = nothing : 인자의 기본값을 설정합니다. 함ㅅ를 호출할 때 intcon 인자에 명시적으로 값을 지정하지 않으면 이 인자의 값은 자동으로 nothing이 됩니다.

이 패턴을 사용하면 사용자가 해당 제약 조건이 없는 경우 그 인자를 함수 호출 시 완전히 생략할 수 있습니다.
"""
function julia_intlinprog(f::AbstractVector{Float64}, intcon::Union{AbstractArray{Int}, Nothing}=nothing;   # 정수 변수 인덱스 벡터
                          binary_var_indices::Union{AbstractVector{Int}, Nothing}=nothing,
                          A::Union{AbstractMatrix{Float64}, Nothing}=nothing,
                          b::Union{AbstractVector{Float64}, Nothing}=nothing,
                          Aeq::Union{AbstractMatrix{Float64}, Nothing}=nothing,
                          beq::Union{AbstractVector{Float64}, Nothing}=nothing,
                          lb::Union{AbstractVector{Float64}, Nothing}=nothing,
                          ub::Union{AbstractVector{Float64}, Nothing}=nothing,
                          optimizer=GLPK.Optimizer,
                          sense::MOI.OptimizationSense=MOI.MIN_SENSE,
                          solver_attributes::Dict{String, Any}=Dict{String, Any}())

    n = length(f)
    model = Model(optimizer)
    for (name, value) in solver_attributes
        set_optimizer_attribute(model, name, value)
    end
    
    # 변수 정의
    @variable(model, x[1:n])

    # 정수 / 이진 제약 설정 (수정됨)
    if intcon !== nothing
        for i in intcon
            if !(1 <= i <= n) # <-- 인덱스 유효성 검사 조건문
                error("Invalid integer variable index : $i. Index must be between 1 and $n.") # <-- 오류 메시지 수정
            end
            # 유효한 인덱스라면 여기서 set_integer를 호출합니다. (if 블록 밖으로 이동)
            set_integer(x[i])
        end
    end

    if binary_var_indices !== nothing
        for i in binary_var_indices
            if !(1 <= i <= n) # <-- 인덱스 유효성 검사 조건문
                error("Invalid binary variable index : $i. Index must be between 1 and $n.") # <-- 오류 메시지 수정
            end
            # 유효한 인덱스라면 여기서 set_binary를 호출합니다. (if 블록 밖으로 이동)
            set_binary(x[i])
        end
    end

    # 나머지 제약 및 목적 함수는 LP와 동일 (이전 코드와 동일)
    if lb !== nothing
        @constraint(model, con_lb[i=1:n], x[i] >= lb[i])
    end
    if ub !== nothing
        @constraint(model, con_ub[i=1:n], x[i] <= ub[i])
    end
    if A !== nothing && b !== nothing
        @constraint(model, con_ineq, A*x .<= b)
    end
    if Aeq !== nothing && beq !== nothing
        @constraint(model, con_eq, Aeq*x .== beq) # 주의: MATLAB Aeq*x = beq 는 등식 제약이므로 .== 이 맞습니다. 이전 코드의 <= 는 오류입니다.
    end

    @objective(model, sense, dot(f, x))

    optimize!(model)

    # MIP 결과는 정수 해로 반환됩니다.
    if termination_status(model) == MOI.OPTIMAL
        return (
            solution=value.(x),
            objective_value=objective_value(model), # obejctive_value 오타 수정
            termination_status=termination_status(model))
    else
        return (
            solution=nothing,
            objective_value=nothing, # obejctive_value 오타 수정
            termination_status=termination_status(model)
        )
    end
end