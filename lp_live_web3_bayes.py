import os
import math
from dotenv import load_dotenv
from web3 import Web3
from scipy.stats import norm

# =============================
# CONFIG
# =============================
TIME_HORIZONS = [14, 30, 60, 90, 180]  # days
BAYESIAN_DO_NOTHING_PRIOR = 0.40

# =============================
# LOAD ENV + CONNECT TO ETH
# =============================
load_dotenv()
INFURA_KEY = os.getenv("INFURA_KEY")
if not INFURA_KEY:
    raise SystemExit("❌ INFURA_KEY not found in .env")

RPC_URL = f"https://mainnet.infura.io/v3/{INFURA_KEY}"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    raise SystemExit("❌ Ethereum connection failed")

# =============================
# USER INPUT
# =============================
token_symbol = input("Enter token symbol (e.g., ETH, USDC): ").upper()
paired_token = input("Enter paired token (e.g., USDC, WETH): ").upper()

manual_bounds = input("Do you want to manually enter price bounds? (y/n): ").lower() == "y"

if manual_bounds:
    LOWER_PRICE = float(input(f"Enter lower price for {token_symbol}/{paired_token}: "))
    UPPER_PRICE = float(input(f"Enter upper price for {token_symbol}/{paired_token}: "))
else:
    LOWER_PRICE = float(input(f"Suggested lower price for {token_symbol}/{paired_token} (e.g., 2800): "))
    UPPER_PRICE = float(input(f"Suggested upper price for {token_symbol}/{paired_token} (e.g., 3600): "))

annual_vol = float(input("Enter annualized volatility (0-1, e.g., 0.75): "))

# =============================
# FETCH LIVE PRICE FROM UNISWAP V3
# =============================
POOL_ADDRESS = Web3.to_checksum_address(input("Enter Uniswap V3 pool address for your pair: "))

POOL_ABI = [
    {
        "inputs": [],
        "name": "slot0",
        "outputs": [
            {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"internalType": "int24", "name": "tick", "type": "int24"},
            {"internalType": "uint16", "name": "observationIndex", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinality", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinalityNext", "type": "uint16"},
            {"internalType": "uint8", "name": "feeProtocol", "type": "uint8"},
            {"internalType": "bool", "name": "unlocked", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

pool = w3.eth.contract(address=POOL_ADDRESS, abi=POOL_ABI)
slot0 = pool.functions.slot0().call()
sqrt_price_x96 = slot0[0]

current_price = 1 / ((sqrt_price_x96 / 2**96) ** 2) * (10 ** (6 - 18))

# =============================
# CALCULATE IN-RANGE PROBABILITIES
# =============================
in_range_probs = {}
for days in TIME_HORIZONS:
    T = days / 365
    sigma = annual_vol
    mu = -0.5 * sigma**2 * T
    std = sigma * math.sqrt(T)
    z_lower = (math.log(LOWER_PRICE / current_price) - mu) / std
    z_upper = (math.log(UPPER_PRICE / current_price) - mu) / std
    in_range_probs[days] = norm.cdf(z_upper) - norm.cdf(z_lower)

# =============================
# BAYESIAN UPDATE
# =============================
likelihood = in_range_probs[14]  # 14-day probability as evidence
posterior = (BAYESIAN_DO_NOTHING_PRIOR * likelihood) / (
    BAYESIAN_DO_NOTHING_PRIOR * likelihood + (1 - BAYESIAN_DO_NOTHING_PRIOR) * (1 - likelihood)
)

# =============================
# OUTPUT
# =============================
print("\nLP OPERATING SYSTEM - LIVE EVALUATION")
print("-----------------------------------")
print(f"Token Pair: {token_symbol}/{paired_token}")
print(f"Live Price: ${round(current_price, 2)}")
print(f"Price Range: ${LOWER_PRICE} → ${UPPER_PRICE}\n")

print("In-Range Probabilities:")
for days, prob in in_range_probs.items():
    print(f"{days} days: {round(prob*100,2)}%")

print(f"\nBayesian Posterior (LP Justified): {round(posterior,3)}")

if posterior < 0.5:
    print("SYSTEM DECISION: ❌ DO NOTHING")
    print("Rationale: Evidence insufficient to overcome inactivity bias.")
else:
    print("SYSTEM DECISION: ✅ LP ALLOWED")
    print("Rationale: Probability-weighted stability justifies LP exposure.")

print("\nCORE RULES:")
print("- LPs are not traders.")
print("- Doing nothing has positive expected value.")
print("- Optimize for survival and regret minimization, not APR.\n")
