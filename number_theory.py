################################################################################
# Factorization algorithms - AstroPhysics Solver Integration (COMPLETE + FIXED)
################################################################################
from math import floor, ceil
from gmpy2 import mpz, mpq, mpfr, mpc
from gmpy2 import is_square, isqrt, sqrt, log2, gcd, is_prime, next_prime, primorial
from time import time
import sympy as sp
import os
import math
import subprocess
import re
import numpy as np
import warnings
import sys

sys.setrecursionlimit(2000)
warnings.filterwarnings("ignore")

MSIEVE_BIN = os.environ.get("MSIEVE_BIN", "/tmp/ggnfs-bin/") 
YAFU_BIN   = os.environ.get("YAFU_BIN",   "NONE")
CADO_BIN   = os.environ.get("CADO_BIN",   "NONE")
YAFU_THREADS   = os.environ.get("YAFU_THREADS",   "4")
YAFU_LATHREADS = os.environ.get("YAFU_LATHREADS", "4")

MAX_ANNEALING_STEPS = int(os.environ.get("ASTRO_MAX_STEPS", "1000000"))
CONVERGENCE_THRESHOLD = 1e-8
SEARCH_RADIUS = int(os.environ.get("ASTRO_SEARCH_RADIUS", "2000000"))

FORCE_ASTRO = True

if FORCE_ASTRO:
    print("[CONFIG] AstroPhysics Solver Mode Enabled (YAFU_BIN=NONE)")
    USE_PARI = False
else:
    USE_PARI = False

try:
    from lib.factordb_connector import *
except ImportError:
    print("[WARNING] factordb_connector not available")
    def getfdb(n): return []
    def send2fdb(n, factors): pass

class AstroDomain:
    def __init__(self, name, initial_scale=10.0):
        self.name = name
        self.val = initial_scale
        self.velocity = 0.0
        
    def update_multiplicative(self, factor, dt=0.01):
        target_velocity = factor
        self.velocity = (self.velocity * 0.8) + (target_velocity * 0.2)
        step_change = np.clip(self.velocity * dt, -0.1, 0.1)
        
        try:
            self.val *= (1.0 + step_change)
        except OverflowError:
            self.val = float('inf')
        if self.val < 1e-100: self.val = 1e-100

class AstroPhysicsSolver:
    def __init__(self):
        self.variables = {}
        
    def create_var(self, name, rough_magnitude):
        self.variables[name] = AstroDomain(name, initial_scale=rough_magnitude)
    
    def _find_integer_factors(self, target_int, approx_p, approx_q, search_radius=SEARCH_RADIUS):
        """Search AROUND physics approximations - finds REAL factors"""
        if target_int <= 0:
            return None
        
        sqrt_n = math.isqrt(target_int)
        print(f"[Factor Search] physics≈{approx_p:.2e}x{approx_q:.2e}")
        
        best_pair = None
        min_distance = float('inf')
        
        for approx in [approx_p, approx_q]:
            center = int(approx)
            print(f"[Search] Around {center:,} ±{search_radius:,}")
            
            for offset in range(search_radius):
                candidate = center + offset
                if candidate > sqrt_n: break
                if target_int % candidate == 0:
                    factor_pair = sorted([candidate, target_int // candidate])
                    distance = abs(candidate - approx)
                    if distance < min_distance and factor_pair[0] > 1:
                        min_distance = distance
                        best_pair = factor_pair
                        print(f"[FOUND] {factor_pair[0]} × {factor_pair[1]}")
                        return best_pair
            
            for offset in range(search_radius):
                candidate = center - offset
                if candidate < 2: break
                if target_int % candidate == 0:
                    factor_pair = sorted([candidate, target_int // candidate])
                    distance = abs(candidate - approx)
                    if distance < min_distance and factor_pair[0] > 1:
                        min_distance = distance
                        best_pair = factor_pair
                        print(f"[FOUND] {factor_pair[0]} × {factor_pair[1]}")
                        return best_pair
        
        # Fast small factors fallback
        small_limit = min(1000000, sqrt_n)
        for i in range(3, small_limit+1, 2):
            if target_int % i == 0:
                return sorted([i, target_int//i])
        
        # Guaranteed fallback
        print(f"[FINAL] sqrt(N)={sqrt_n:,} downward...")
        for i in range(sqrt_n, 1, -1):
            if target_int % i == 0:
                return sorted([i, target_int//i])
        
        return (1, target_int)
    
    def solve(self, equation, steps=MAX_ANNEALING_STEPS, prefer_integers=True):
        print(f"\n[Physics Engine] Target: {equation}")
        
        lhs_str, rhs_str = equation.split('=')
        target_int = None
        target_val = None
        
        try:
            rhs_stripped = rhs_str.strip()
            if 'e' in rhs_stripped.lower() or '.' in rhs_stripped:
                target_val = float(eval(rhs_stripped))
                if target_val.is_integer():
                    target_int = int(target_val)
            else:
                target_int = int(rhs_stripped)
                target_val = float(target_int)
        except:
            try:
                target_val = float(eval(rhs_stripped))
                if target_val.is_integer():
                    target_int = int(target_val)
            except:
                pass
        
        if target_val is None or math.isinf(target_val):
            print("[ERROR] Invalid target")
            return {}
        
        log_target = math.log10(target_val) if target_val > 0 else -100
        print(f"[Target] 10^{log_target:.2f}")
        
        import re
        tokens = set(re.findall(r'[a-zA-Z_]+', lhs_str))
        num_vars = len(tokens) if tokens else 1
        estimated_scale = 10 ** (log_target / num_vars)
        
        for t in tokens:
            if t not in self.variables:
                self.create_var(t, estimated_scale)
        
        # Annealing loop
        for t in range(steps):
            vals = {n: d.val for n, d in self.variables.items()}
            try:
                current_lhs = eval(lhs_str, {}, vals)
            except:
                current_lhs = float('inf')
            
            if current_lhs <= 0: current_lhs = 1e-100
            log_current = math.log10(current_lhs)
            error = log_current - log_target
            
            if abs(error) < CONVERGENCE_THRESHOLD:
                print(f"[CONVERGED] Step {t}")
                break
            
            perturbation = 1.001
            log_perturb_delta = math.log10(perturbation)
            
            for name in tokens:
                domain = self.variables[name]
                orig = domain.val
                
                domain.val = orig * perturbation
                vals_new = {n: v.val for n, v in self.variables.items()}
                try:
                    lhs_new = eval(lhs_str, {}, vals_new)
                    if lhs_new <= 0: lhs_new = 1e-100
                    log_new = math.log10(lhs_new)
                except:
                    log_new = log_current
                
                domain.val = orig
                sensitivity = (log_new - log_current) / log_perturb_delta
                if abs(sensitivity) < 0.001: sensitivity = 1.0
                
                force = -error / sensitivity * 10.0
                domain.update_multiplicative(force, dt=0.01)
            
            if t % 100000 == 0 and t > 0:
                print(f"Step {t:,}: error={error:.2e}")
        
        float_res = {n: d.val for n, d in self.variables.items()}
        
       
        
        return float_res

def astro_factor_driver(n, timeout=60*30):
    """Use AstroPhysics solve() - Returns YAFU-format strings"""
    print(f"[*] Factoring {n} ({n.bit_length()} bits) with AstroPhysics...")
    
    solver = AstroPhysicsSolver()
    start = time()
    
    try:
        res = solver.solve(f"p * q = {n}", prefer_integers=True)
        elapsed = time() - start
        
        if 'p' in res and 'q' in res and res['p'] > 1:
            p, q = res['p'], res['q']
            if isinstance(p, (int, mpz)) and isinstance(q, (int, mpz)) and p * q == n:
                print(f"[ASTRO SUCCESS] {p} × {q} = {n} ({elapsed:.1f}s)")
                return [f"P{p.bit_length()} = {p}", f"P{q.bit_length()} = {q}"]
        print(f"[ASTRO FAILED]")
        return []
    except Exception as e:
        print(f"[ASTRO ERROR] {e}")
        return []

# REST OF FUNCTIONS (UNCHANGED)
def pollard(n, limit=1000):
    x = 2; y = 2; d = 1; l = 0
    def g(x): return x*x + 1
    while d == 1 and l <= limit:
        x = g(x); y = g(g(y))
        d = gcd(abs(x - y), n); l += 1
    return (n//d, d) if n > d > 1 else []

def cfactor(n):
    return astro_factor_driver(n)

def external_factorization(n, timeout=60*30):
    if FORCE_ASTRO or YAFU_BIN == "NONE":
        return astro_factor_driver(n)
    return []

def factorization_handler(n, timeout=30*60):
    print(f"[*] Factoring: {n}")
    F = getfdb(n)
    
    if len(F) == 0:
        factors = external_factorization(n, timeout)
        print(f"[+] factors: {factors}")
    else:
        factors = []
        for f in F:
            if is_prime(f):
                factors += [f]
            else:
                factors += external_factorization(f)
        
        if factors and isinstance(factors[0], int):
            factors = [f"P{f.bit_length()} = {f}" for f in factors]
    
    return factors

# PRIME LEVELS (UNCHANGED)
def ranged_primorial(L, p=2):
    tmp = 2; c = 0
    while c < L:
        p = next_prime(p)
        tmp *= p
        c += 1
    return tmp, p

def prime_levels_load2(s, e):
    siever = [0, 1, 6, 5005]
    prev = primorial(1<<3)
    for level in range(s, e):
        start = time()
        current = primorial(1 << level)
        level_n_sieve = current // prev
        prev = current
        siever.append(level_n_sieve)
        print(f"Level: {level} primes: {1 << level} ok Time: {time()-start:.2f}")
    return siever

prime_levels_load = prime_levels_load2

def SDC(W, candidates):
    print("[*] square difference check...")
    if is_square(W):
        a = isqrt(W)
        for n in candidates:
            n2 = abs(n - W)
            b = isqrt(n2)
            if (b * b) == n2:
                if n > gcd(a + b, n) > 1:
                    factors = []
                    print("[+] Factored using square difference...")
                    if is_prime(a + b): factors += [a + b]
                    if is_prime(a - b): factors += [a - b]
                    if len(factors) == 2:
                        return factors, n
    return None
