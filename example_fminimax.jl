include("julia_fminimax.jl")
# --- 목적 함수 정의 (JuMP 표현식 또는 제약 조건 내에 직접) ---
# 이 부분은 사용자의 실제 목적 함수 f_i(x)에 따라 달라집니다.
# 각 목적 함수를 @NLexpression 또는 @expression으로 정의합니다.
# 예시: f1(x) = x[1]^2, f2(x) = (x[2]-1)^2
@NLexpression(model, obj1, x[1]^2)
@NLexpression(model, obj2, (x[2]-1)^2)

# 목적 함수 표현식을 벡터로 저장하여 반복문에서 사용
objective_expressions = [obj1, obj2]


# --- 미니맥스 제약 조건 추가: f_i(x) <= gamma for all i ---
for i in 1:n_obj
    # objective_expressions[i]가 비선형이면 @NLconstraint, 선형이면 @constraint 사용 가능
    # 비선형일 가능성이 있다면 @NLconstraint를 사용하는 것이 안전합니다.
    @NLconstraint(model, objective_expressions[i] <= gamma)
end


# --- 기타 제약 조건 추가 ---
# 사용자의 선형 및 비선형 제약 조건을 여기에 추가해야 합니다.

# 선형 부등식 제약 조건 A * x <= b
# if A !== nothing && b !== nothing
#     @constraint(model, A * x .<= b)
# end

# 선형 등식 제약 조건 Aeq * x == beq
# if Aeq !== nothing && beq !== nothing
#     @constraint(model, Aeq * x .== beq)
# end

# 비선형 제약 조건 c(x) <= 0 및 ceq(x) == 0
# @NLconstraint 매크로를 사용하여 직접 정의합니다.
# 예시 비선형 부등식: @NLconstraint(model, x[1]^2 + x[2]^2 <= 1.0)
# 예시 비선형 등식: @NLconstraint(model, x[1] * x[2] == 0.5)


# --- 목적 함수 설정 (gamma 최소화) ---
@objective(model, Min, gamma)


# --- 모델 최적화 실행 ---
println("최적화 시작...")
optimize!(model)
println("최적화 완료.")


# --- 결과 확인 ---
# 비선형 솔버는 MOI.OPTIMAL 또는 MOI.LOCALLY_SOLVED 상태를 반환할 수 있습니다.
if termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    println("\n================== 최적화 결과 ==================")
    println("최적 상태: ", termination_status(model))
    println("해결 상태: ", primal_status(model))
    println("최적 변수 x: ", value.(x))
    println("최적 gamma 값: ", value(gamma))

    # 최적 해에서의 각 목적 함수 값 계산
    optimal_obj_values = value.(objective_expressions)
    println("최적 해에서의 목적 함수 값: ", optimal_obj_values)

    # 최적 gamma 값과 목적 함수 값의 비교 (모든 f_i(x) <= gamma 인지 확인)
    # gamma의 정의에 따라 최적gamma는 최적해에서의 목적 함수 값들 중 최댓값과 같아야 합니다.
    println("최적 해에서의 목적 함수 값들 중 최댓값: ", maximum(optimal_obj_values))
    println("gamma 값: ", value(gamma))


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