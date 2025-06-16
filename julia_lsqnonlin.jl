using Optim
using LinearAlgebra
using ForwardDiff
# using Plots # 이 예제에서는 시각화가 필요 없으므로 주석 처리하거나 삭제합니다.

"""
julia_lsqnonlin_optim (Optim.jl을 사용한 비선형 최소 제곱)

Optim.jl 공식 문서(https://www.juliapackages.com/p/optim)를 기반으로,
residual_vector_func(p) 함수가 반환하는 잔차 벡터의 제곱합 sum(residual_vector_func(p).^2)을
최소화하는 파라미터 p를 찾습니다.

Optim.jl을 사용하여 일반적인 스칼라 목적 함수 최소화 문제로 변환하여 해결합니다.
ForwardDiff.jl을 사용하여 목적 함수의 그래디언트를 자동으로 계산합니다.

인자:
- residual_vector_func: 파라미터 벡터 p를 유일한 인자로 받아 잔차 벡터를 반환하는 함수.
                        이 함수는 최적화에 필요한 모든 추가 데이터(예: xdata, ydata)를
                        정의 시의 스코프에서 캡처해야 합니다.
                        (예: `(p) -> y_data .- model(x_data, p)`)
                        -> Optim.jl에 직접 전달될 스칼라 목적 함수와 그 그래디언트 계산에 사용됩니다.
- p0: 파라미터 벡터의 초기 추정값 (AbstractVector{Float64}).
- lb: 파라미터의 하한 벡터 (AbstractVector{Float64} 또는 nothing). 기본값은 nothing (제약 없음).
- ub: 파라미터의 상한 벡터 (AbstractVector{Float64} 또는 nothing). 기본값은 nothing (제약 없음).
      *주의*: 경계 조건을 사용하려면 lb와 ub를 모두 제공해야 합니다.
             Fminbox(Optim.LBFGS())와 같은 경계 조건 지원 알고리즘이 필요합니다.
- algorithm: 사용할 Optim.jl 알고리즘 객체. 기본값은 `nothing`이며,
             경계 조건 유무에 따라 Optim.LBFGS() 또는 Fminbox(Optim.LBFGS())가 자동 선택됩니다.
             사용자가 명시적으로 다른 Optim.jl 알고리즘을 지정할 수 있습니다.
             (문서의 "Algorithms" 섹션 참고)
- options: Optim.Options 객체. 최적화 알고리즘의 설정 조정. 기본값은 `Optim.Options()`.
           (문서의 "Options" 섹션 참고)

반환 값:
Named Tuple 형태의 결과. (문서의 "Result types" 참고)
(
    solution: 최적화된 파라미터 벡터 (MATLAB의 x에 해당). Optim.minimizer(result)로 얻음.
    objective_value: 최적화된 파라미터에서의 잔차 제곱합 (MATLAB의 fval 또는 resnorm에 해당). Optim.minimum(result)로 얻음.
    is_converged: 최적화 수렴 기준 만족 여부 (Bool). Optim.converged(result)로 얻음.
    optim_result: Optim.jl의 전체 결과 객체. (상세 정보: 반복 횟수, 평가 횟수, 다른 수렴 기준 등).
)
"""
function julia_lsqnonlin_optim(residual_vector_func, p0::AbstractVector{Float64};
                               lb::Union{AbstractVector{Float64}, Nothing}=nothing,
                               ub::Union{AbstractVector{Float64}, Nothing}=nothing,
                               algorithm=nothing, # Default algorithm logic handled below
                               options=Optim.Options())

    n = length(p0)

    # 입력 유효성 검사
    if lb !== nothing && length(lb) != n
        error("하한 벡터 (lb)의 길이가 초기 추정값 (p0)의 길이와 일치해야 합니다.")
    end
    if ub !== nothing && length(ub) != n
        error("상한 벡터 (ub)의 길이가 초기 추정값 (p0)의 길이와 일치해야 합니다.")
    end
     if (lb !== nothing && ub === nothing) || (lb === nothing && ub !== nothing)
         error("경계 조건을 사용하려면 하한 (lb)과 상한 (ub)을 모두 제공해야 합니다.")
     end
     if lb !== nothing && ub !== nothing && any(lb .> ub)
         error("하한 (lb) 값은 상한 (ub) 값보다 작거나 같아야 합니다.")
     end

    # determine the algorithm to use based on bounds if not specified
    used_algorithm = algorithm
    if used_algorithm === nothing
        if lb !== nothing && ub !== nothing
            used_algorithm = Optim.Fminbox(Optim.LBFGS()) # Use Fminbox(LBFGS()) as a robust default for bounds
            println("알고리즘이 지정되지 않아 경계 조건에 적합한 Fminbox(Optim.LBFGS())를 기본값으로 사용합니다.")
        else
            used_algorithm = Optim.LBFGS() # Default to LBFGS for unconstrained problems
             println("알고리즘이 지정되지 않아 경계 조건이 없어 Optim.LBFGS()를 기본값으로 사용합니다.")
        end
    end

    # Define the scalar objective function: sum of squares of residuals
    obj(p) = sum(abs2, residual_vector_func(p))

    # Define the gradient of the scalar objective function using ForwardDiff
    grad_obj!(G, p) = ForwardDiff.gradient!(G, obj, p)

    # Optim.optimize 함수 호출
    if lb !== nothing && ub !== nothing
        # 경계 조건이 있는 경우 (Fminbox 또는 다른 경계 지원 알고리즘 사용)
        optim_result = Optim.optimize(obj, grad_obj!, lb, ub, p0, used_algorithm, options)
    else
        # 경계 조건이 없는 경우 (LBFGS 또는 다른 비제약 알고리즘 사용)
        optim_result = Optim.optimize(obj, grad_obj!, p0, used_algorithm, options)
    end

    # 결과 추출
    p_fitted = Optim.minimizer(optim_result)
    fval = Optim.minimum(optim_result)
    is_converged_status = Optim.converged(optim_result)

    # 결과를 Named Tuple 형태로 반환
    return (
        solution=p_fitted,
        objective_value=fval,
        is_converged=is_converged_status,
        optim_result=optim_result
    )
end