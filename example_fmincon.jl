include("julia_fmincon.jl")

# --- 예제 문제 정의 및 fmincon_jump 호출 ---
# 이제 fmincon_jump 함수가 fun과 nonlcon 인자를 사용합니다.

# 1. 목적 함수 정의: f(x) = (x1 - 1)^2 + (x2 - 2)^2
fun_ex2 = x -> (x[1]-1)^2 + (x[2]-2)^2;

# 2. 초기 추측값
x0_ex2 = [0.0, 0.0]

# 3. 선형 제약 조건 (A*x <= b)
# x1 + x2 <= 4
A_ex2 = [1.0 1.0] # 행렬
b_ex2 = [4.0]     # 벡터

# 4. 선형 등식 제약 조건 (Aeq*x == beq) - 없음
Aeq_ex2 = nothing
beq_ex2 = nothing

# 5. 변수 경계 (lb <= x <= ub)
# 0 <= x1 <= 3
# 0 <= x2 <= 3
lb_ex2 = [0.0, 0.0]
ub_ex2 = [3.0, 3.0]

# 6. 비선형 부등식 제약 조건 (nonlcon(x) <= 0)
# x1^2 + x2^2 <= 5  =>  x1^2 + x2^2 - 5 <= 0
nonlcon_ex2 = x -> x[1]^2 + x[2]^2 - 5.0; # 스칼라 값을 반환하는 함수

# fmincon_jump 함수 호출
minx_ex2, minf_ex2, status_ex2 = fmincon_jump(
    fun_ex2, # 정의된 목적 함수 전달
    x0_ex2;
    A=A_ex2,
    b=b_ex2,
    Aeq=Aeq_ex2,
    beq=beq_ex2,
    lb=lb_ex2,
    ub=ub_ex2,
    nonlcon=nonlcon_ex2 # 정의된 비선형 제약 함수 전달
)

println("\n--- 예제 2 결과 ---")
println("Status: ", status_ex2)
println("Optimal x: ", minx_ex2)
println("Minimum objective value: ", minf_ex2)

# 예상 결과는 여전히 LOCALLY_SOLVED 상태와 함께 최적해가 출력될 것입니다.
# 이 최적해는 제약 조건을 만족하면서 목적 함수 값을 최소화하는 점입니다.