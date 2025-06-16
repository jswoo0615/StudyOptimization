include("julia_fgoalattain.jl")
# --- 목적 함수 정의 (JuMP 표현식 또는 제약 조건 내에 직접) ---
# 이 부분은 사용자의 실제 목적 함수 f_i(x)에 따라 달라집니다.
# 만약 목적 함수가 비선형이라면 @NLexpression 매크로를 사용합니다.
# 예시: f1(x) = x[1]^2, f2(x) = (x[2]-1)^2
@NLexpression(model, obj1, x[1]^2)
@NLexpression(model, obj2, (x[2]-1)^2)
# 목적 함수 표현식을 벡터로 저장
objective_expressions = [obj1, obj2]

# --- 목표 달성 제약 조건 추가 ---
for i in 1:n_obj
    if weight[i] > 0
        # f_i(x) - w_i * gamma <= g_i
        @NLconstraint(model, objective_expressions[i] - weight[i] * gamma <= goal[i])
    elseif weight[i] < 0
        # g_i - f_i(x) <= |w_i| * gamma  =>  f_i(x) - g_i >= -|w_i| * gamma => f_i(x) - g_i >= weight[i] * gamma
        @NLconstraint(model, objective_expressions[i] - weight[i] * gamma >= goal[i])
    else # weight[i] == 0
        # f_i(x) == g_i
        @NLconstraint(model, objective_expressions[i] <= goal[i])
        @NLconstraint(model, objective_expressions[i] >= goal[i])
    end
end

# --- 기타 제약 조건 추가 ---
# 선형 부등식 제약 조건 A * x <= b
if A !== nothing && b !== nothing
    @constraint(model, A * x .<= b)
end

# 선형 등식 제약 조건 Aeq * x == beq
if Aeq !== nothing && beq !== nothing
    @constraint(model, Aeq * x .== beq)
end

# 비선형 제약 조건 (사용자가 필요에 따라 직접 추가)
# 예: @NLconstraint(model, x[1]^2 + x[2]^2 <= 1.0)

# --- 목적 함수 설정 (gamma 최소화) ---
@objective(model, Min, gamma)

# --- 모델 최적화 실행 ---
println("최적화 시작...")
optimize!(model)
println("최적화 완료.")

# --- 결과 확인 ---
# 최적화 성공 상태는 MOI.OPTIMAL 또는 MOI.LOCALLY_SOLVED 등을 포함할 수 있습니다.
if termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    println("\n================== 최적화 결과 ==================")
    println("최적 상태: ", termination_status(model))
    println("해결 상태: ", primal_status(model))
    println("최적 변수 x: ", value.(x))
    println("최적 gamma 값: ", value(gamma))

    # 최적 해에서의 각 목적 함수 값 계산
    optimal_obj_values = value.(objective_expressions)
    println("최적 해에서의 목적 함수 값: ", optimal_obj_values)

    # 각 목표 달성 상태 확인 (f_i(x) - g_i)
    println("최적 해에서의 목표 달성 편차 (f_i(x) - g_i): ", optimal_obj_values .- goal)

else
    println("\n================== 최적화 결과 ==================")
    println("최적 해를 찾지 못했거나 다른 상태로 종료되었습니다.")
    println("종료 상태: ", termination_status(model))
    println("해결 상태: ", primal_status(model))
    # 부분적으로 얻은 값 확인 (가능하다면)
    if has_values(model)
         println("부분 최적 변수 x: ", value.(x))
         println("부분 최적 gamma 값: ", value(gamma))
    end
end