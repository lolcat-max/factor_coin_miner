################################################################################
# Factorization algorithms - AstroPhysics Solver Integration
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

MSIEVE_BIN = os.environ.get("MSIEVE_BIN", "/tmp/ggnfs-bin/") 
YAFU_BIN   = os.environ.get("YAFU_BIN",   "NONE")
CADO_BIN   = os.environ.get("CADO_BIN",   "NONE")
YAFU_THREADS   = os.environ.get("YAFU_THREADS",   "4")
YAFU_LATHREADS = os.environ.get("YAFU_LATHREADS", "4")

# AstroPhysics Solver Configuration
MAX_ANNEALING_STEPS = int(os.environ.get("ASTRO_MAX_STEPS", "500000"))
CONVERGENCE_THRESHOLD = 1e-10
SEARCH_RADIUS = int(os.environ.get("ASTRO_SEARCH_RADIUS", "200000"))

# Force AstroPhysics mode when YAFU is disabled
FORCE_ASTRO = True
#Factoring libraries setup
if FORCE_ASTRO:
    print("[CONFIG] AstroPhysics Solver Mode Enabled (YAFU_BIN=NONE)")
    USE_PARI = False
    pari_cfactor = None
else:
    USE_PARI = False
   

# Import factordb connector
try:
    from lib.factordb_connector import *
except ImportError:
    print("[WARNING] factordb_connector not available")
    def getfdb(n):
        return []
    def send2fdb(n, factors):
        pass


class AstroDomain:
    """Astrophysics-inspired variable domain for numerical optimization"""
    def __init__(self, name, initial_scale=10.0):
        self.name = name
        self.val = float(initial_scale)
        self.velocity = 0.0
        
    def update_multiplicative(self, force, dt=0.01):
        """Update using multiplicative physics with damping"""
        target_velocity = float(force)
        self.velocity = (self.velocity * 0.8) + (target_velocity * 0.2)
        step_change = max(min(self.velocity * dt, 0.1), -0.1)
        
        try:
            self.val *= (1.0 + step_change)
        except OverflowError:
            self.val = float('inf')
        
        if self.val < 1e-100:
            self.val = 1e-100


class AstroPhysicsSolver:
    """Numerical factorization using astrophysics-inspired annealing"""
    
    def __init__(self):
        self.variables = {}
    
    def factor(self, N, max_steps=MAX_ANNEALING_STEPS):
        """
        Factor semiprime N into p * q using physics-inspired optimization
        Returns (p, q) tuple or None if factorization fails
        """
        if N <= 1:
            return None
        
        # Quick checks first
        if N % 2 == 0:
            return (2, N // 2)
        
        # Check if it's a perfect square
        sqrt_n = int(isqrt(N))
        if sqrt_n * sqrt_n == N:
            if is_prime(sqrt_n):
                return (sqrt_n, sqrt_n)
        
        # Initialize variables in log space
        log_target = math.log10(float(N))
        estimated_scale = 10 ** (log_target / 2)
        
        self.variables = {
            'x': AstroDomain('x', estimated_scale),
            'y': AstroDomain('y', estimated_scale)
        }
        
        print(f"[ASTRO] Starting physics optimization for N={N} ({N.bit_length()} bits)")
        print(f"[PARAMS] Initial scale: {estimated_scale:.2e}, Max steps: {max_steps:,}")
        
        # Annealing loop
        converged = False
        best_error = float('inf')
        best_x = estimated_scale
        best_y = estimated_scale
        
        for step in range(max_steps):
            x_val = self.variables['x'].val
            y_val = self.variables['y'].val
            
            try:
                current_product = x_val * y_val
                if current_product <= 0:
                    current_product = 1e-100
                log_current = math.log10(current_product)
            except (ValueError, OverflowError):
                log_current = -100
            
            error = log_current - log_target
            
            # Track best approximation
            if abs(error) < abs(best_error):
                best_error = error
                best_x = x_val
                best_y = y_val
            
            # Check convergence
            if abs(error) < CONVERGENCE_THRESHOLD:
                converged = True
                print(f"[CONVERGED] Step {step}: error={error:.2e}")
                break
            
            # Compute sensitivities (gradient-free)
            perturbation = 1.001
            log_perturb_delta = math.log10(perturbation)
            
            for var_name in ['x', 'y']:
                domain = self.variables[var_name]
                orig_val = domain.val
                
                # Perturb and measure
                domain.val = orig_val * perturbation
                x_new = self.variables['x'].val
                y_new = self.variables['y'].val
                
                try:
                    product_new = x_new * y_new
                    log_new = math.log10(product_new) if product_new > 0 else -100
                except (ValueError, OverflowError):
                    log_new = log_current
                
                # Restore
                domain.val = orig_val
                
                # Compute sensitivity
                sensitivity = (log_new - log_current) / log_perturb_delta
                if abs(sensitivity) < 0.001:
                    sensitivity = 1.0
                
                # Apply correction force
                force = -error / sensitivity
                force *= 10.0  # Gain
                
                domain.update_multiplicative(force, dt=0.01)
            
            # Progress report
            if step % 50000 == 0 and step > 0:
                print(f"  Step {step:,}: x={x_val:.6e}, y={y_val:.6e}, error={error:.2e}")
        
        if not converged:
            print(f"[ASTRO] Using best approximation after {max_steps:,} steps (error={best_error:.2e})")
            self.variables['x'].val = best_x
            self.variables['y'].val = best_y
        
        # Extract integer factors
        approx_x = self.variables['x'].val
        approx_y = self.variables['y'].val
        
        factors = self._snap_to_integers(N, approx_x, approx_y)
        
        if factors:
            p, q = factors
            print(f"[ASTRO SUCCESS] Found factors: {p} × {q} = {N}")
            return (p, q)
        else:
            print(f"[ASTRO FAILED] Could not snap to integer factors")
            return None
    
    def _snap_to_integers(self, N, approx_x, approx_y):
        """Find exact integer factors near approximate solutions"""
        sqrt_n = int(isqrt(N))
        
        # Ensure x <= y convention
        if approx_x > approx_y:
            approx_x, approx_y = approx_y, approx_x
        
        # Search around the approximation
        center = int(approx_x)
        start = max(2, center - SEARCH_RADIUS)
        end = min(center + SEARCH_RADIUS, sqrt_n + 1)
        
        print(f"[INTEGER SNAP] Searching [{start:,}, {end:,}] around {center:,}")
        
        # Search around the estimated factor
        for candidate in range(center, end):
            if N % candidate == 0:
                complement = N // candidate
                if candidate <= complement:
                    return (candidate, complement)
        
        for candidate in range(center - 1, start - 1, -1):
            if N % candidate == 0:
                complement = N // candidate
                if candidate <= complement:
                    return (candidate, complement)
        
        # Fallback: search from sqrt downward
        print(f"[FALLBACK] Broadening search from sqrt(N)={sqrt_n:,}")
        search_limit = min(100000, sqrt_n // 10)
        for candidate in range(sqrt_n, max(2, sqrt_n - search_limit), -1):
            if N % candidate == 0:
                complement = N // candidate
                return (min(candidate, complement), max(candidate, complement))
        
        return None


def pollard(n, limit=1000):
    x = 2
    y = 2
    d = 1
    l = 0
    def g(x):
      return x*x + 1
    while d == 1 and l <= limit:
        x = g(x) 
        y = g(g(y)) 
        d = gcd(abs(x - y), n)
        l += 1
    if n > d > 1:
      return n//d, d
    else:
      return []
    

def astro_factor_driver(n, timeout=60*30):
    """Use AstroPhysics solver for factorization"""
    print("[*] Factoring %d with AstroPhysics solver..." % n)
    
    solver = AstroPhysicsSolver()
    start = time()
    
    try:
        factors = solver.factor(n)
        elapsed = time() - start
        
        if factors:
            p, q = factors
            # Verify
            if p * q == n and is_prime(p) and is_prime(q):
                print(f"[ASTRO SUCCESS] Factored in {elapsed:.2f}s")
                # CRITICAL: Return strings in YAFU format "P<bits> = <value>"
                return [f"P{p.bit_length()} = {p}", f"P{q.bit_length()} = {q}"]
            else:
                print(f"[ASTRO FAILED] Invalid factors or not both prime")
                # Return partial result if multiplication is correct
                if p * q == n:
                    return [f"C{p.bit_length()} = {p}", f"C{q.bit_length()} = {q}"]
                return []
        else:
            print(f"[ASTRO FAILED] No factors found in {elapsed:.2f}s")
            return []
            
    except Exception as e:
        print(f"[ASTRO ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def cadonfs_factor_driver(n):
  global CADO_BIN
  print("[*] Factoring %d with cado-nfs..." % n)
  tmp = []
  proc = subprocess.Popen([ CADO_BIN, str(n)], stdout=subprocess.PIPE)
  for line in proc.stdout:
    line = line.rstrip().decode("utf8")
    if re.search("\d+",line):
      tmp += [int(x) for x in line.split(" ")]
  return tmp


def msieve_factor_driver(n):
  global MSIEVE_BIN
  print("[*] Factoring %d with msieve..." % n) 
  import subprocess, re, os
  tmp = []
  proc = subprocess.Popen([MSIEVE_BIN,"-s","/tmp/%d.dat" % n,"-t","8","-v",str(n)],stdout=subprocess.PIPE)
  for line in proc.stdout:
    line = line.rstrip().decode("utf8")
    if re.search("factor: ",line):
      tmp += [int(line.split()[2])]
  os.system("rm %d.dat" % n)
  return tmp


def yafu_factor_driver(n, timeout = 60*30):
  global YAFU_BIN
  print("[*] Factoring %d with yafu..." % n)
  import subprocess, re, os
  tmp = []
  try:
     parse = subprocess.run([YAFU_BIN,"-one","-plan", "custom","-pretest_ratio","0.32", "-threads",YAFU_THREADS,"-lathreads",YAFU_LATHREADS, "-ggnfs_dir", MSIEVE_BIN, "-xover", "160", "-snfs_xover", "150",  str(n)], timeout=timeout, stdout=subprocess.PIPE)
  except subprocess.TimeoutExpired:
     print("Timeout: ", timeout, " Seconds. Moving on to next candidate")
     return [ "C" + str( n.bit_length() ) + " = " + str(n) ]

  parse = [ line for line in parse.stdout.decode('utf-8').split("\n") if "=" in line ]
  tmp = []
  flag=False

  for line in parse:
      if "Total factoring" in line:
          flag = True
          continue
      if "ans" in line:
          continue

      if flag:
          tmp.append(line)

  return tmp


def cfactor(n):
    if not USE_PARI or pari_cfactor is None:
        print("[*] PARI not available, using AstroPhysics solver...")
        return astro_factor_driver(n)  # This already returns strings
  
    print("[*] pari factoring: %d..." % n)
    f0 = pari_cfactor(n)
    factors = [int(a) for a in f0[0]]
    # Convert to string format for compatibility
    return [f"P{f.bit_length()} = {f}" for f in factors]
    
    print("[*] pari factoring: %d..." % n)
    f0 = pari_cfactor(n)
    factors  = [int(a) for a in f0[0]]
    return factors


def external_factorization(n, timeout=60*30):
  factors = []
  
  # Force AstroPhysics solver when YAFU is disabled
  if FORCE_ASTRO or YAFU_BIN == "NONE":
      print("[*] Using AstroPhysics solver (YAFU disabled)...")
      factors = astro_factor_driver(n, timeout)
      return factors
  
  # Otherwise use PARI/YAFU
  if USE_PARI:
      factors = cfactor(n)
      # Convert integer factors to string format for compatibility
      if factors and isinstance(factors[0], int):
          factors = [f"P{f.bit_length()} = {f}" for f in factors]
  else:
      factors = yafu_factor_driver(n, timeout)

  return factors


def factorization_handler(n, timeout=30*60):
  print("[*] Factoring:", n)

  # Check FactorDB first (optional, currently disabled)
  F = []

  if len(F) == 0:
    factors = external_factorization(n, timeout)
    print("[+] factors: %s" % str(factors))
  else:
    print("fdb got: %d factors: %s" % (len(F), str(F)))
    factors = []
    for f in F:
      if is_prime(f):
        factors += [f]
      else:
        factors += external_factorization(f)
        
  # Ensure factors are in string format
  if factors and isinstance(factors[0], int):
      factors = [f"P{f.bit_length()} = {f}" for f in factors]
  
  return factors


def ranged_primorial(L, p = 2):
  tmp = 2
  c = 0
  while c < L:
    p = next_prime(p)
    tmp *= p
    c += 1
  return tmp, p


def prime_levels_load0(s, e):
  siever = [ 0, 1, 6, 5005 ]
  prev, last_p = ranged_primorial(8)
  for level in range(s, e):
    siever.append(ranged_primorial(1 << level, p = last_p))
    print("Level %d ok" % level)
  return siever


def prime_levels_load1(s, e):
  siever = [ 0, 1, 6, 5005 ]
  prev = sp.primorial( 1<<3, False)
  for level in range(s, e):
    current       = sp.primorial( 1 << level, False)
    level_n_sieve = current//prev
    prev          = current
    siever.append(  level_n_sieve )
    print("Level %d ok" % level)
  return siever


def prime_levels_load2(s, e):
  siever = [ 0, 1, 6, 5005 ]
  prev = primorial( 1<<3)
  for level in range(s, e):
    start = time()
    current       = primorial( 1 << level)
    level_n_sieve = current//prev
    prev          = current
    siever.append(  level_n_sieve )
    print("Level: %d primes: %d  ok Time: %f" % (level, (1 << level), time()-start ))
  return siever


def prime_levels_load_timing():
  t0 = time.time()
  prime_levels_load0(4, 16)
  t1 = time.time()
  print(t1-t0)
  prime_levels_load1(4, 16)
  t2 = time.time()
  print(t2-t1)
  prime_levels_load2(4, 16)
  t3 = time.time()
  print(t3-t2)


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
          if is_prime(a + b):
            factors += [a + b]
          if is_prime(a - b):
            factors += [a - b]
          if len(factors) == 2:
            return factors, n
  return None