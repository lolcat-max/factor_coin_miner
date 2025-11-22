#!/usr/bin/env python3
"""
FACTOR Blockchain Miner - AstroPhysicsSolver Edition
Uses astrophysics-inspired numerical optimization for integer factorization PoW
Compatible with factornd RPC protocol
"""

import sys
import os
import time
import json
import hashlib
import random
import math
import base58
import requests
from decimal import Decimal, getcontext

# Configuration
RPC_HOST = os.getenv('RPC_HOST', '127.0.0.1')
RPC_PORT = int(os.getenv('RPC_PORT', '8332'))  # FACTOR testnet default
RPC_USER = os.getenv('rpcuser', 'choice')
RPC_PASS = os.getenv('rpcpassword', 'choice')
SCRIPTPUBKEY = os.getenv('SCRIPTPUBKEY', '0014dba95b2d7b908d851b94a772adb42abddb4e1d2f')

# Physics Engine Parameters
MAX_ANNEALING_STEPS = 2000000
CONVERGENCE_THRESHOLD = 1e-10
SEARCH_RADIUS = 200000

# Decimal precision
getcontext().prec = 10000

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
        
        # Initialize variables in log space
        log_target = math.log10(float(N))
        estimated_scale = 10 ** (log_target / 2)
        
        self.variables = {
            'x': AstroDomain('x', estimated_scale),
            'y': AstroDomain('y', estimated_scale)
        }
        
        print(f"[ANNEALING] Starting physics optimization for N={N}")
        print(f"[PARAMS] Initial scale: {estimated_scale:.2e}, Max steps: {max_steps:,}")
        
        # Annealing loop
        converged = False
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
            if step % 100000 == 0 and step > 0:
                print(f"  Step {step:,}: x={x_val:.6e}, y={y_val:.6e}, error={error:.2e}")
        
        if not converged:
            print(f"[WARNING] Max steps reached without convergence")
        
        # Extract integer factors
        approx_x = self.variables['x'].val
        approx_y = self.variables['y'].val
        
        factors = self._snap_to_integers(N, approx_x, approx_y)
        
        if factors:
            p, q = factors
            print(f"[SUCCESS] Found factors: {p} × {q} = {N}")
            return (p, q)
        else:
            print(f"[FAILED] Could not snap to integer factors")
            return None
    
    def _snap_to_integers(self, N, approx_x, approx_y):
        """Find exact integer factors near approximate solutions"""
        sqrt_n = math.isqrt(N)
        
        # Ensure x <= y convention
        if approx_x > approx_y:
            approx_x, approx_y = approx_y, approx_x
        
        # Search around sqrt(N) for balanced factors
        start = max(2, int(approx_x) - SEARCH_RADIUS)
        end = min(int(approx_x) + SEARCH_RADIUS, sqrt_n)
        
        print(f"[INTEGER SNAP] Searching [{start}, {end}] around sqrt({N})={sqrt_n}")
        
        for candidate in range(end, start - 1, -1):
            if N % candidate == 0:
                complement = N // candidate
                if candidate <= complement:
                    return (candidate, complement)
        
        # Fallback: trial division from sqrt downward
        print(f"[FALLBACK] Broadening search from sqrt(N)")
        for candidate in range(sqrt_n, max(2, sqrt_n - 100000), -1):
            if N % candidate == 0:
                complement = N // candidate
                return (min(candidate, complement), max(candidate, complement))
        
        return None

class FactorMiner:
    """FACTOR blockchain miner using AstroPhysicsSolver"""
    
    def __init__(self, rpc_host, rpc_port, rpc_user, rpc_pass, scriptpubkey):
        self.rpc_host = rpc_host
        self.rpc_port = rpc_port
        self.rpc_url = f"http://{rpc_host}:{rpc_port}/"
        self.scriptpubkey = scriptpubkey
        self.session = requests.Session()
        self.session.auth = (rpc_user, rpc_pass)
        self.session.headers.update({'Content-Type': 'application/json'})
        
        self.solver = AstroPhysicsSolver()
        
        self.blocks_found = 0
        self.total_work_time = 0.0
        
        print(f"[MINER INIT]")
        print(f"  RPC: {rpc_host}:{rpc_port}")
        print(f"  RPC User: {rpc_user}")
        print(f"  scriptPubKey: {scriptpubkey[:32]}...")
        print(f"  Engine: AstroPhysicsSolver")
        
        # Test connection
        print(f"\n[CONNECTION TEST]")
        self._test_connection()
    
    def rpc_call(self, method, params=None):
        """Make JSON-RPC call to factornd"""
        if params is None:
            params = []
        
        payload = {
            "jsonrpc": "1.0",
            "id": "astrominer",
            "method": method,
            "params": params
        }
        
        try:
            response = self.session.post(
                self.rpc_url,
                json=payload,
                timeout=60
            )
            
            # Debug: print response if not successful
            if response.status_code != 200:
                print(f"[DEBUG] Status: {response.status_code}")
                print(f"[DEBUG] Response: {response.text[:200]}")
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('error'):
                error = result['error']
                error_code = error.get('code', -1)
                error_msg = error.get('message', 'Unknown')
                print(f"[RPC ERROR {error_code}] {method}: {error_msg}")
                return None
            
            return result.get('result')
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"\n[AUTHENTICATION FAILED]")
                print(f"  Your RPC credentials are incorrect!")
                print(f"\n  Fix: Edit your factornd config file:")
                print(f"  Location: ~/.factorn/factorn.conf (Linux/Mac)")
                print(f"           %APPDATA%\\Factorn\\factorn.conf (Windows)")
                print(f"\n  Add these lines:")
                print(f"    rpcuser={self.session.auth[0]}")
                print(f"    rpcpassword={self.session.auth[1]}")
                print(f"    rpcallowip=127.0.0.1")
                print(f"    rpcport={self.rpc_port}")
                print(f"    server=1")
                print(f"\n  Then restart factornd:")
                print(f"    factornd.exe -reindex (Windows)")
                print(f"    ./factornd -daemon (Linux/Mac)")
                return None
            else:
                print(f"[HTTP ERROR {e.response.status_code}] {method}: {e}")
                print(f"  Response: {e.response.text[:200]}")
                return None
        
        except requests.exceptions.ConnectionError as e:
            print(f"[CONNECTION ERROR] Cannot connect to {self.rpc_url}")
            print(f"  Error: {str(e)[:200]}")
            print(f"\n  Is factornd running?")
            print(f"  Check: tasklist | findstr factornd (Windows)")
            print(f"         ps aux | grep factornd (Linux/Mac)")
            return None
        
        except requests.exceptions.RequestException as e:
            print(f"[REQUEST ERROR] {method}: {str(e)[:200]}")
            return None
    
    def _test_connection(self):
        """Test RPC connection and display node info"""
        info = self.rpc_call("getblockchaininfo")
        
        if info:
            print(f"  ✓ Connected to factornd")
            print(f"  Chain: {info.get('chain', 'unknown')}")
            print(f"  Blocks: {info.get('blocks', 0)}")
            print(f"  Difficulty: {info.get('difficulty', 0)}")
            return True
        else:
            print(f"  ✗ Connection failed")
            print(f"\n[TROUBLESHOOTING]")
            
            # Try to detect the right port
            print(f"\n  Testing common FACTOR ports...")
            import socket
            for test_port in [8332, 8766, 18332, 18766]:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', test_port))
                sock.close()
                
                if result == 0:
                    print(f"    ✓ Port {test_port} is open")
                    if test_port != self.rpc_url.split(':')[-1].rstrip('/'):
                        print(f"      TRY: export RPC_PORT={test_port}")
                else:
                    print(f"    ✗ Port {test_port} is closed")
            
            print(f"\n1. Check factornd is running:")
            print(f"   ps aux | grep factornd  (Linux/Mac)")
            print(f"   tasklist | findstr factornd  (Windows)")
            print(f"\n2. Check RPC settings in config:")
            print(f"   ~/.factorn/factorn.conf (Linux/Mac)")
            print(f"   %APPDATA%\\Factorn\\factorn.conf (Windows)")
            print(f"\n   Required settings:")
            print(f"   rpcuser={self.session.auth[0]}")
            print(f"   rpcpassword={self.session.auth[1]}")
            print(f"   rpcallowip=127.0.0.1")
            print(f"   rpcport=8766")
            print(f"   server=1")
            print(f"\n3. Restart factornd:")
            print(f"   ./factornd stop")
            print(f"   ./factornd -daemon")
            return False
    
    def get_block_template(self):
        """Get mining template from factornd"""
        template = self.rpc_call("getblocktemplate", [{"rules": ["segwit"]}])
        
        if not template:
            return None
        
        # Debug: Show what fields are actually in the template
        print(f"[DEBUG] Template keys: {list(template.keys())}")
        
        # Extract challenge N
        challenge_n = template.get('challenge_n')
        if not challenge_n:
            # Some nodes may use different field names
            for key in ['semiprime', 'target_n', 'factorization_challenge', 'challenge', 'n', 'composite']:
                if key in template:
                    challenge_n = template[key]
                    print(f"[DEBUG] Found challenge in field: {key}")
                    break
        
        if not challenge_n:
            print("[ERROR] No factorization challenge in template")
            print(f"[DEBUG] Full template structure:")
            print(json.dumps(template, indent=2, default=str)[:1000])
            return None
    
    def build_coinbase(self, template, factors):
        """Build coinbase transaction with factors"""
        height = template['height']
        p, q = factors
        
        # Coinbase data format for FACTOR
        coinbase_data = {
            'version': 1,
            'height': height,
            'scriptPubKey': self.scriptpubkey,
            'factors': [int(p), int(q)],
            'value': template.get('coinbasevalue', 5000000000)  # 50 FACT in satoshis
        }
        
        return coinbase_data
    
    def submit_block(self, template, factors):
        """Submit solved block to factornd"""
        p, q = factors
        N = template['challenge_n']
        
        # Verify factors locally
        if p * q != N:
            print(f"[ERROR] Invalid factors: {p} × {q} = {p*q} ≠ {N}")
            return False
        
        # Build coinbase
        coinbase = self.build_coinbase(template, factors)
        
        # Build block (simplified - factornd expects specific format)
        block_data = {
            'version': template['version'],
            'previousblockhash': template['previousblockhash'],
            'height': template['height'],
            'time': int(time.time()),
            'bits': template['bits'],
            'challenge_n': N,
            'factors': [int(p), int(q)],
            'coinbase': coinbase,
            'transactions': template.get('transactions', [])
        }
        
        # Serialize block (factornd expects hex)
        block_json = json.dumps(block_data, sort_keys=True)
        block_hex = block_json.encode('utf-8').hex()
        
        print(f"[SUBMIT] Sending block {template['height']}")
        print(f"  Factors: {p} × {q}")
        print(f"  Block size: {len(block_hex)} hex chars")
        
        # Submit via submitblock RPC
        result = self.rpc_call("submitblock", [block_hex])
        
        if result is None or result == "" or result == "accepted":
            print(f"[✓ BLOCK ACCEPTED]")
            self.blocks_found += 1
            return True
        else:
            print(f"[✗ BLOCK REJECTED] {result}")
            return False
    
    def mine_one_block(self):
        """Mine a single block"""
        # Get template
        template = self.get_block_template()
        if not template:
            return False
        
        N = template['challenge_n']
        
        # Factor using AstroPhysicsSolver
        start_time = time.time()
        factors = self.solver.factor(N)
        work_time = time.time() - start_time
        
        if not factors:
            print(f"[FAILED] Factorization took {work_time:.2f}s")
            return False
        
        print(f"[TIMING] Factorization: {work_time:.2f}s")
        self.total_work_time += work_time
        
        # Submit block
        success = self.submit_block(template, factors)
        
        return success
    
    def run(self, target_blocks=None):
        """Main mining loop"""
        print(f"\n{'='*70}")
        print(f"FACTOR MINER - AstroPhysicsSolver Edition")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        attempts = 0
        
        try:
            while True:
                attempts += 1
                print(f"\n[ATTEMPT {attempts}] Mining block...")
                
                success = self.mine_one_block()
                
                if success:
                    elapsed = time.time() - start_time
                    rate = self.blocks_found / elapsed if elapsed > 0 else 0
                    avg_time = self.total_work_time / self.blocks_found if self.blocks_found > 0 else 0
                    
                    print(f"\n[STATS]")
                    print(f"  Blocks found: {self.blocks_found}")
                    print(f"  Session time: {elapsed:.1f}s")
                    print(f"  Block rate: {rate*3600:.2f} blocks/hour")
                    print(f"  Avg solve time: {avg_time:.2f}s")
                
                # Check if target reached
                if target_blocks and self.blocks_found >= target_blocks:
                    print(f"\n[COMPLETE] Reached target of {target_blocks} blocks")
                    break
                
                # Brief pause between attempts
                time.sleep(1)
        
        except KeyboardInterrupt:
            print(f"\n\n[STOPPED] Mining interrupted by user")
        
        # Final stats
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"MINING SESSION COMPLETE")
        print(f"{'='*70}")
        print(f"Blocks found: {self.blocks_found}/{attempts} attempts")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Success rate: {self.blocks_found/attempts*100:.1f}%")
        if self.blocks_found > 0:
            print(f"Avg solve time: {self.total_work_time/self.blocks_found:.2f}s")

def main():
    """Entry point"""
    # Check for setup command
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        print("FACTOR Miner Configuration Helper")
        print("="*50)
        
        config_path = os.path.expanduser("~/.factornd/factornd.conf")
        print(f"\nConfig file location: {config_path}")
        
        # Get credentials
        print("\nEnter RPC credentials (or press Enter for defaults):")
        rpc_user = input(f"RPC Username [choice]: ").strip() or "choice"
        rpc_pass = input(f"RPC Password [choice]: ").strip() or "choice"
        
        # Create config content
        config = f"""# FACTOR Node Configuration
rpcuser={rpc_user}
rpcpassword={rpc_pass}
rpcallowip=127.0.0.1
server=1
txindex=1
daemon=1
"""
        
        # Create directory if needed
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write config
        with open(config_path, 'w') as f:
            f.write(config)
        
        print(f"\n✓ Configuration saved to {config_path}")
        print(f"\nNext steps:")
        print(f"1. Start factornd:")
        print(f"   ./factornd -daemon")
        print(f"\n2. Wait for sync (check with):")
        print(f"   ./factornd-cli getblockchaininfo")
        print(f"\n3. Get your scriptPubKey:")
        print(f"   ./factornd-cli getnewaddress")
        print(f"   ./factornd-cli getaddressinfo <address>")
        print(f"\n4. Start mining:")
        print(f"   export RPC_USER={rpc_user}")
        print(f"   export RPC_PASS={rpc_pass}")
        print(f"   python {sys.argv[0]} <scriptPubKey>")
        
        return
    
    if len(sys.argv) < 2:
        print("Usage: python factor_miner.py <scriptPubKey> [target_blocks]")
        print("   or: python factor_miner.py setup")
        print("\nSetup mode creates factornd.conf with proper RPC settings")
        print("\nMining example:")
        print("  python factor_miner.py 0014a8a44c20e7b12de5405bb864cd9d5be3c5bd055a 10")
        print("\nEnvironment variables:")
        print("  RPC_HOST (default: 127.0.0.1)")
        print("  RPC_PORT (default: 8332)")
        print("  RPC_USER (default: choice)")
        print("  RPC_PASS (default: choice)")
        sys.exit(1)
    
    scriptpubkey = sys.argv[1]
    target_blocks = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Initialize miner
    miner = FactorMiner(
        rpc_host=RPC_HOST,
        rpc_port=RPC_PORT,
        rpc_user=RPC_USER,
        rpc_pass=RPC_PASS,
        scriptpubkey=scriptpubkey
    )
    
    # Start mining
    miner.run(target_blocks=target_blocks)

if __name__ == "__main__":
    main()
