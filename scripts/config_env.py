"""Print the env block of a server configuration as KEY=VALUE lines (used by serve.sh).
Values already set in the environment override the config; server-side sampling defaults are appended to OMLX_EXTRA.
Usage: python scripts/config_env.py configs/q6_omlx_cand5.json"""
import json, os, sys

env = {k: os.environ.get(k, v) for k, v in json.load(open(sys.argv[1]))["env"].items()}
env["OMLX_EXTRA"] = ", ".join(x for x in (env.get("OMLX_EXTRA"), '"temperature": 0.6, "top_p": 0.95, "top_k": 20') if x)
print("\n".join(f"{k}={v}" for k, v in env.items()))
