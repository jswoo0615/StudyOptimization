include("julia_quadprog.jl")
# --- 3. quadprog 예제 ---
println("\n--- Running quadprog equivalent example ---")
# Minimize 0.5 * x' * [2 0; 0 2] * x + [0, 0]' * x  s.t. [-1 -1]*x <= -1  (i.e., x1^2 + x2^2 s.t. x1+x2 >= 1)
H_qp = [2.0 0.0; 0.0 2.0]
f_linear_qp = [0.0, 0.0]
A_qp = [-1.0 -1.0]
b_qp = [-1.0]
qp_result = julia_quadprog(H_qp, f_linear_qp, A=A_qp, b=b_qp)
println("Result: ", qp_result)
