include("julia_linprog.jl")
"""
Minimize -x1 -2x2
Subject to : 
    x1 + x2 <= 1
    2x1 + x2 <= 1.5
    x1 >= 0, x2 >= 0
"""

f = [-1.0, -2.0]            # 목적 함수 계수 (Minimize f'x)

A = [1.0 1.0;               # x1 + x2 <= 1
    2.0 1.0]                # 2x1 + x2 <= 1.5

b = [1.0, 1.5]

Aeq = nothing               # 등식 제약 없음
beq = nothing

lb = [0.0, 0.0 ]            # x1 >= 0, x2 >= 0
ub = nothing                # 상한 제약 없음

# -------- 추가 : 솔버 옵션 설정 --------
# 솔버 (GLPK)의 메시지 출력 레벨을 설정하는 옵션을 딕셔너리에 담습니다.
# `msg_lev`는 GLPK에서 메시지 레벨을 설정하는 옵션 이름입니다.
# GLPK.MSG_LEV_ALL, GLPK.MSG_LEV_INFO, GLPK.MSG_LEV_DBG 등 다양한 레벨이 있습니다.
# GLPK.MSG_LEV_ALL은 가장 상세한 출력을 제공합니다.
glpk_options = Dict{String, Any}("msg_lev"=>GLPK.GLP_MSG_ALL)


# 정의한 julia_linprog 함수를 사용하여 문제 풀이
# minimize가 기본이므로 sense는 MOI.MIN_SENSE 그대로 사용합니다.
lp_result = julia_linprog(f, A=A, b=b, lb=lb, solver_attributes=glpk_options)

# 결과 출력
println("\n--- Solving Example LP using julia_linprog ---")
if lp_result.termination_status == MOI.OPTIMAL
    println("Solution : x = $(lp_result.solution)")
    println("Objective value : $(lp_result.objective_value)")
else
    println("Optimization Failed Status : $(lp_result.termination_status)")
end
println("---------------------------------------------")

"""
GLPK Simplex Optimizer 5.0                              -> 문제를 푼 솔버의 종류와 버전. GLPK 솔버가 심플렉스 (Simplex) 알고리즘을 사용하여 문제를 풀고 있음을 나타냅니다.
4 rows, 2 columns, 6 non-zeros                          -> 솔버에 전달된 LP 문제의 크기와 구조에 대한 정보입니다.
                                                        -> 2 columns : 결정 변수의 갯수 (x1, x2)입니다.
                                                        -> 4 rows : 제약 조건의 갯수입니다. 예제 문제의 제약 조건은 
                                                           x1 + x2 <= 1,
                                                           2x1 + x2 <= 1.5,
                                                           x1 >= 0, x2 >= 0 (이는 -x1 <= 0, -x2 <=0 으로 변환 가능)
                                                           총 4개의 부등식 제약기 있으므로 4 rows와 일치합니다.
                                                        -> 6 non-zeros : 제약 조건 행렬에 있는 0이 아닌 계수의 총 갯수입니다.
*     0: obj =   0.000000000e+00 inf =   0.000e+00 (2)  -> inf = 0.000e+00 : 현재 해의 비실행 가능성 (infeasiblilty) : 여기서는 제약 조건을 모두 만족하는 실행 가능한 해라는 뜻입니다.
*     2: obj =  -2.000000000e+00 inf =   0.000e+00 (0)  
OPTIMAL LP SOLUTION FOUND

--- Solving Example LP using julia_linprog ---
Solution : x = [0.0, 1.0]
Objective value : -2.0
"""
