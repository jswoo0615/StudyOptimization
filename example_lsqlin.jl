include("julia_lsqlin.jl")

# --- julia_lsqlin 사용 예제 ---
# 이 예제는 Wrapper 함수 외부에서 실행되어야 합니다.

# 예제: 간단한 제약 조건이 있는 선형 최소 제곱 문제
# min ||C*x - d||^2
# s.t. A*x <= b
#      Aeq*x = beq
#      lb <= x <= ub

# C, d 행렬/벡터 정의
C_ex = [1.0 2.0; 3.0 4.0; 5.0 6.0]
d_ex = [1.0, 2.0, 7.0]

# 선형 부등식 제약 A*x <= b (옵션)
A_ex = [1.0 -1.0; -1.0 -1.0]
b_ex = [1.0, -2.0]

# 선형 등식 제약 Aeq*x = beq (옵션)
Aeq_ex = [1.0 1.0]
beq_ex = [3.0]

# 변수 x의 하한/상한 제약 lb <= x <= ub (옵션)
lb_ex = [0.0, 0.0]
ub_ex = [10.0, 10.0]

# julia_lsqlin 함수 호출

println("\n--- julia_lsqlin 실행 결과 (모든 제약 조건 포함) ---")
# 모든 인자를 명시적으로 전달 (타입 확인 용이)
result_lsqlin = julia_lsqlin(C_ex, d_ex, A=A_ex, b=b_ex, Aeq=Aeq_ex, beq=beq_ex, lb=lb_ex, ub=ub_ex)

# 결과 출력
println("최적화된 변수 (solution): ", result_lsqlin.solution)
println("최소 잔차 제곱합 (objective_value): ", result_lsqlin.objective_value)
println("종료 상태: ", result_lsqlin.termination_status)
println("프라이멀 상태: ", result_lsqlin.primal_status)
println("듀얼 상태: ", result_lsqlin.dual_status)
# println("JuMP 모델: ", result_lsqlin.jump_model) # 필요시 전체 모델 확인

# 다른 제약 조건 조합으로 호출 가능
println("\n--- julia_lsqlin 실행 결과 (경계 조건만 포함) ---")
result_lsqlin_bounds = julia_lsqlin(C_ex, d_ex, lb=lb_ex, ub=ub_ex)
println("최적화된 변수 (solution): ", result_lsqlin_bounds.solution)
println("최소 잔차 제곱합 (objective_value): ", result_lsqlin_bounds.objective_value)
println("종료 상태: ", result_lsqlin_bounds.termination_status)

println("\n--- julia_lsqlin 실행 결과 (제약 조건 없음) ---")
result_lsqlin_unconstrained = julia_lsqlin(C_ex, d_ex) # Unconstrained LLS
println("최적화된 변수 (solution): ", result_lsqlin_unconstrained.solution)
println("최소 잔차 제곱합 (objective_value): ", result_lsqlin_unconstrained.objective_value)
println("종료 상태: ", result_lsqlin_unconstrained.termination_status)

# Unconstrained LLS can be verified with matrix division C \ d
println("\n--- 비제약 LLS 직접 계산 (C \\ d) 결과 ---")
direct_solution = C_ex \ d_ex
direct_objective = sum(abs2, C_ex * direct_solution - d_ex)
println("직접 계산된 해: ", direct_solution)
println("직접 계산된 목적 함수 값: ", direct_objective)