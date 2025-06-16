include("julia_fsolve.jl")

# --- julia_fsolve 사용 예제 ---
# 이 예제는 Wrapper 함수 외부에서 실행되어야 합니다.

# 예제: 간단한 비선형 방정식 시스템 찾기
# F(x) = 0
# F1(x1, x2) = x1^2 + x2^2 - 1 = 0  (단위원)
# F2(x1, x2) = x2 - x1^2 = 0       (포물선)

# 시스템 F(x) = 0 를 계산하는 인플레이스 함수 정의 (fcn!(F, x) 형태)
function circle_parabola_system!(F, x)
    F[1] = x[1]^2 + x[2]^2 - 1
    F[2] = x[2] - x[1]^2
    return nothing # 인플레이스 함수는 보통 nothing을 반환
end

# 해 찾기를 위한 초기 추정값
x0_example = [0.8, 0.8] # 양수 해 근처

# julia_fsolve 함수 호출
println("\n--- julia_fsolve 실행 결과 ---")
result_fsolve = julia_fsolve(circle_parabola_system!, x0_example; ftol=1e-10, iterations=100)

# 결과 출력
println("찾은 해 (solution): ", result_fsolve.solution)
# 이제 residuals_at_solution는 수동 계산된 잔차 벡터입니다.
println("해에서의 잔차 (residuals_at_solution): ", result_fsolve.residuals_at_solution)
println("수렴 여부: ", result_fsolve.is_converged)
println("NLsolve 결과 상세 정보: ", result_fsolve.nlsolve_result) # 전체 결과 객체를 출력하여 상세 정보 확인 가능

# 다른 초기 추정값으로 호출 (음수 해 근처)
println("\n--- julia_fsolve 실행 결과 (다른 초깃값) ---")
x0_example_neg = [-0.8, 0.8]
result_fsolve_neg = julia_fsolve(circle_parabola_system!, x0_example_neg; ftol=1e-10, iterations=100)

# 결과 출력
println("찾은 해 (solution): ", result_fsolve_neg.solution)
println("해에서의 잔차 (residuals_at_solution): ", result_fsolve_neg.residuals_at_solution)
println("수렴 여부: ", result_fsolve_neg.is_converged)
println("NLsolve 결과 상세 정보: ", result_fsolve_neg.nlsolve_result)