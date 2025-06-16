using JuMP
using GLPK                  # LP 솔버, 원한다면 다른 솔버 (HiGHS.Optimizer 등)로 변경 가능합니다.
using LinearAlgebra         # dot 함수 사용
using MathOptInterface      # MOI.OptimizationSense 등을 명시적으로 사용하기 위해 추가
using Cbc
using HiGHS
include("julia_intlinprog.jl")
println("\n--- Running intlinprog equivalent example (Knapsack) ---")
# Maximize 6x1 + 10x2 + 10x3
# s.t. 2x1 + 3x2 + 4x3 <= 5
# x1, x2, x3 are Bin (0 or 1)
f_mip = [-6.0, -10.0, -10.0] # Maximize Z is equivalent to Minimize -Z
A_mip = [2.0 3.0 4.0]
b_mip = [5.0]
binary_indices_mip = [1, 2, 3] # 모든 변수가 정수 (이진은 정수의 부분집합)
lb_mip = [0.0, 0.0, 0.0]
ub_mip = [1.0, 1.0, 1.0]

options = Dict{String, Any}("msg_lev"=>GLPK.GLP_MSG_ALL)
# options = Dict{String, Any}("output_flag"=>true)

# Binary 변수는 set_integer 대신 set_binary를 사용해야 하지만,
# julia_intlinprog 함수는 Intcon만 받도록 되어 있으므로 사용자가 Binary임을 인지하고 있어야 합니다.
# 더 일반적인 intlinprog 래퍼는 Binary 변수 인자도 받을 수 있습니다.
# 여기서는 단순화하여 모든 정수 변수를 Intcon으로 전달합니다.
# 이 예제는 Binary 문제이므로, 내부적으로 set_binary가 호출되거나
# MIP 솔버가 Int 제약과 변수 경계 (0<=x<=1)를 보고 Binary로 처리합니다.
# GLPK 솔버는 Int 제약과 0, 1 경계를 보면 Binary로 인식합니다.

mip_result = julia_intlinprog(f_mip,
                               binary_var_indices=binary_indices_mip, # <-- 인자명 확인
                               A=A_mip,
                               b=b_mip,
                               lb=lb_mip, # <-- lb 전달
                               ub=ub_mip, # <-- ub 전달
                            #    optimizer=HiGHS.Optimizer,
                               optimizer=GLPK.Optimizer,
                               sense=MOI.MIN_SENSE,
                               solver_attributes=options) # Minimize -Z


# 결과를 Maximize 관점에서 해석하여 출력
if mip_result.termination_status == MOI.OPTIMAL
    println("Result (Maximize): Solution = ", mip_result.solution, ", Objective value = ", -mip_result.objective_value)
else
    println("Result: ", mip_result)
end