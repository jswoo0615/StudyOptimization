include("julia_lsqnonlin.jl")

# --- julia_lsqnonlin_optim 사용 예제 (요청하신 목적 함수와 초깃값 사용) ---
# 이 예제는 Wrapper 함수 외부에서 실행되어야 합니다.

# 요청하신 목적 함수 f(x) 정의
# 이 함수는 lsqnonlin의 입력처럼 파라미터 벡터 x를 받아 잔차 벡터를 반환합니다.
function f_user_provided(x)
    r1 = (x[1] - 1)^2 + (x[2] - 2)^2 - 1
    r2 = x[1] + x[2] - 3
    return [r1, r2] # 두 개의 잔차 값을 벡터로 반환
end

# julia_lsqnonlin_optim에 전달할 '잔차 벡터 함수' 정의 (데이터 캡처 없음, f_user_provided 사용)
# 이 함수는 f_user_provided를 단순히 호출하는 Wrapper 역할을 합니다.
# 데이터 캡처가 필요한 문제라면 이 부분에서 외부 데이터를 사용합니다.
residual_vec_func_for_optim(p) = f_user_provided(p)

# 요청하신 초깃값
p_init_example = [0.0, 0.0]

# julia_lsqnonlin_optim 함수 호출 (경계 조건 없음)
println("\n--- Optim.jl (경계 조건 없음) 실행 결과 ---")
# 알고리즘을 지정하지 않으면 LBFGS()가 기본값으로 사용됩니다.
result_optim = julia_lsqnonlin_optim(residual_vec_func_for_optim, p_init_example)

# 결과 출력
println("최적화된 파라미터 (x): ", result_optim.solution)
println("최소 잔차 제곱합 (fval): ", result_optim.objective_value)
println("수렴 여부: ", result_optim.is_converged)
println("Optim.jl 결과 상세 정보: ", result_optim.optim_result)


# julia_lsqnonlin_optim 함수 호출 (경계 조건 포함)
println("\n--- Optim.jl (경계 조건 포함) 실행 결과 ---")
# Copilot 예제에서 사용된 -5, 5 경계 조건을 사용합니다.
lb_example = [-5.0, -5.0]
ub_example = [5.0, 5.0]

# 경계 조건이 있으므로 Fminbox(Optim.LBFGS())가 기본값으로 사용됩니다.
# algorithm=Optim.Fminbox(Optim.LBFGS()) 를 명시적으로 지정해도 됩니다.
result_optim_bounded = julia_lsqnonlin_optim(residual_vec_func_for_optim, p_init_example, lb=lb_example, ub=ub_example)

# 결과 출력
println("경계 조건 하의 최적화된 파라미터 (x): ", result_optim_bounded.solution)
println("경계 조건 하의 최소 잔차 제곱합 (fval): ", result_optim_bounded.objective_value)
println("수렴 여부: ", result_optim_bounded.is_converged)
println("Optim.jl 결과 상세 정보: ", result_optim_bounded.optim_result)