using NLsolve
using LinearAlgebra # similar() 함수 사용

"""
julia_fsolve (비선형 방정식 시스템 해 찾기)

비선형 방정식 시스템 F(x) = 0 의 해(근)를 찾습니다.
MATLAB의 fsolve와 유사한 기능을 제공합니다.

NLsolve.jl 패키지의 nlsolve 함수를 기반으로 합니다.
SolverResults 객체의 .residual 필드에 의존하지 않고 수동으로 잔차를 계산합니다.

인자:
- fcn!: 방정식 시스템 F(x)를 정의하는 인플레이스 함수. 시그니처는 fcn!(F, x) 형태여야 합니다.
        x는 입력 벡터이고, F는 계산 결과 F(x)를 저장할 미리 할당된 출력 벡터입니다.
- x0: 해를 찾기 위한 초기 추정값 벡터 (AbstractVector{Float64}).
- kwargs: NLsolve.nlsolve 함수에 직접 전달될 키워드 인자들.
          일반적인 옵션: ftol, iterations, method, autodiff 등.

반환 값:
Named Tuple 형태의 결과.
(
    solution: 찾은 해 벡터 x. result.zero로 얻음. (MATLAB의 x)
    residuals_at_solution: 찾은 해 x에서의 잔차 벡터 F(x). 수동 계산. (MATLAB의 fval)
    is_converged: 해 찾기 알고리즘이 수렴 기준을 만족했는지 여부 (Bool). NLsolve.converged(result)로 얻음. (MATLAB의 exitflag와 관련)
    nlsolve_result: NLsolve.jl의 전체 결과 객체. (상세 정보: 반복 횟수, 평가 횟수, 수렴 이유 등) (MATLAB의 output)
)
"""
function julia_fsolve(fcn!, x0::AbstractVector{Float64}; kwargs...)

    # NLsolve.nlsolve 함수 호출
    result = nlsolve(fcn!, x0; kwargs...)

    # 결과 추출
    solution = result.zero # 찾은 해 벡터

    # --- 수동 잔차 계산 ---
    # result.residual 필드 대신, 찾은 해(solution)를 원본 함수(fcn!)에 대입하여 잔차를 계산합니다.
    final_residuals = similar(solution) # 잔차를 저장할 벡터를 해의 크기와 타입으로 미리 할당
    fcn!(final_residuals, solution)    # 사용자 정의 함수를 호출하여 final_residuals에 결과를 저장

    is_converged_status = NLsolve.converged(result) # 수렴 여부 확인 (Bool)

    # 결과를 Named Tuple 형태로 반환
    return (
        solution=solution,
        residuals_at_solution=final_residuals, # 수동 계산된 잔차 벡터 반환
        is_converged=is_converged_status,
        nlsolve_result=result # NLsolve.jl의 전체 결과 객체 포함 (상세 정보 확인용)
    )
end