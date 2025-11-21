# bayesian_opt.py - uses scikit-optimize (skopt) to run a toy BO process
import os, json, time
import numpy as np, pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
os.makedirs('bo_outputs', exist_ok=True)
def objective(params):
    lr, hidden = params
    # synthetic objective surface (lower is better)
    score = 0.2 + (np.log10(lr)+3)**2 * 0.02 + (150-hidden)/1000.0
    score += np.random.normal(0, 0.005)
    time.sleep(0.01)
    return score
space = [Real(1e-5, 1e-2, "log-uniform", name='lr'), Integer(32, 256, name='hidden_dim')]
res = gp_minimize(objective, space, n_calls=20, random_state=42)
df = pd.DataFrame(res.x_iters, columns=[s.name for s in space])
df['score'] = res.func_vals
df.to_csv('bo_outputs/bo_results.csv', index=False)
with open('bo_outputs/opt_best.json','w') as f:
    json.dump({'best_params': dict(zip([s.name for s in space], res.x)), 'best_score': float(res.fun)}, f, indent=2)
with open('bo_outputs/optimization_trace.txt','w') as f:
    for i,(x,v) in enumerate(zip(res.x_iters, res.func_vals)):
        f.write(f'iter {i}: params={x} score={v}\n')
print('BO done, results saved in bo_outputs/')
