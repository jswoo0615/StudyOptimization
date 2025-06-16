using JuMP
using Ipopt # 예시 솔버 로드
using LinearAlgebra # 행렬/벡터 연산에 필요할 수 있음
using MathOptInterface # 결과 상태 확인용

# --- 사용자의 문제 데이터 ---
# 결정 변수의 개수 (MATLAB의 x0 길이)
n_vars = 2

# 미니맥스하려는 목적 함수의 개수
n_obj = 2

# 선형 부등식 제약 조건: A * x <= b
A = nothing # 예: [1.0 1.0; -1.0 2.0]
b = nothing # 예: [5.0, 3.0]

# 선형 등식 제약 조건: Aeq * x == beq
Aeq = nothing # 예: [1.0 0.0]
beq = nothing # 예: [1.5]

# 변수 상하한: lb <= x <= ub
# Inf/-Inf를 포함할 수 있으며, 코드가 이를 처리합니다.
lb = [-Inf, -Inf] # 예: [0.0, 0.0]
ub = [Inf, Inf] # 예: [10.0, 10.0]

# 초기 추측값 (선택 사항, 솔버에 따라 사용될 수 있음)
x0 = [0.0, 0.0] # 예시 초기값

# 비선형 제약 조건 (선택 사항)은 코드 내에서 @NLconstraint로 직접 정의합니다.

# --- JuMP 모델 설정 및 최적화 ---

# JuMP 모델 생성 및 솔버 연결
model = Model(Ipopt.Optimizer)

# 결정 변수 x 정의
@variable(model, x[1:n_vars])

# 보조 변수 gamma 정의 (미니맥스 값)
@variable(model, gamma)

# 초기 추측값 설정 (선택 사항)
if x0 !== nothing
    set_start_value.(x, x0)
end
set_start_value(gamma, 0.0) # gamma의 초기값 설정

# 변수 상하한 설정 (Inf/-Inf 처리 포함)
if lb !== nothing
    for i in 1:n_vars
        if !isinf(lb[i])
            set_lower_bound(x[i], lb[i])
        end
    end
end

if ub !== nothing
    for i in 1:n_vars
        if !isinf(ub[i])
            set_upper_bound(x[i], ub[i])
        end
    end
end