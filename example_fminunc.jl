include("julia_fminunc.jl")
# --- 5. fminunc 예제 ---
println("\n--- Running fminunc equivalent example ---")
# Minimize (x-1)^2 + (y-2)^2
f_unc = (x) -> (x[1]-1)^2 + (x[2]-2)^2 # Julia 함수 정의
x0_unc = [0.0, 0.0] # 초기 추측값
unc_result = julia_fminunc(f_unc, x0_unc)
println("Result: ", unc_result)