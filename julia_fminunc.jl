using JuMP
using Optim
"""
fminunc (Unconstrained Nonlinear Programming)
Minimize fun(x)
"""
function julia_fminunc(fun::Function, x0::AbstractVector{Float64};
                       algorithm=Optim.LBFGS(),     # 기본 알고리즘
                       options=Optim.Options())
    # Optim.optimize 함수 사용
    result = Optim.optimize(fun, x0, algorithm, options)

    # if Optim.converged(result)
    #     return (solution=Optim.minimizer(result),
    #             objective_value=Optim.minimum(result),
    #             termination_status=Optim.outer_termination_status(result))
    # else
    #     return (solution=Optim.minimizer(result),
    #             objective_value=Optim.minimum(result),
    #             termination_status=Optim.outer_termination_status(result))
        # Optim은 수렴하지 않아도 마지막 minimizer를 반환할 수 있습니다.
    return (solution=Optim.minimizer(result),
            objective_value=Optim.minimum(result),
            converged=Optim.converged(result),
            result_object=result)
    
end