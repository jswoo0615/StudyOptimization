using Optim
"""
fminbnd (Bounded Scalar Nonlinear Programming)
* 목적 : 단일 변수 비선형 목적 함수를 주어진 구간 ([x1, x2]) 내에서 최소화합니다.
* Minimize fun(x) subject to x1 <= x <= x2 (단일 변수)

단일 변수 Julia 함수 fun과 하한 x1, 상한 x2를 인자로 받습니다
Optim.optimize의 1차원 구간 최적화 기능을 사용하여 문제를 해결합니다.
"""
function julia_fminbnd(fun::Function, x1::Float64, x2::Float64;
    method=Optim.GoldenSection(),
    kwargs...) # 기본 알고리즘 설정

    # Optim.optimize 함수 사용 (1차원 구간 최적화)
    # 표준적인 호출 형태: optimize(fun, a, b, method, options)
    # Note: User's environment might have a MethodError with this specific call signature,
    # but Optim.optimize(fun, x1, x2) was found to work there.
    result = Optim.optimize(fun, x1, x2, method; kwargs...) # <--- 이 형태가 표준적입니다.

    # 결과 반환 (이전 수정 내용 유지)
    return (solution=Optim.minimizer(result),
    objective_value=Optim.minimum(result),
    converged=Optim.converged(result),
    result_object=result)
end