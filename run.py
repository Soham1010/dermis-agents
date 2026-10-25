"""Run the Dermis agents, one process per agent (the Agentverse Inspector can't connect Bureaus).

  python run.py                       all 12 agents: hub on :8000, specialists on :8001-8011
  DERMIS_MAILBOX=1 python run.py      same, reachable from Agentverse / ASI:One through mailboxes
  python run.py --only hub,quadruped  a subset (the hub runs missing specialists itself)
  python run.py --solo hub            one agent in this process
  python run.py --addresses           print every agent's address and port, then exit
"""

import argparse
import logging
import signal
import socket
import subprocess
import sys
import threading

from dermis_agents import agents, config


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def stream(key: str, proc: subprocess.Popen) -> None:
    for line in proc.stdout:
        print(f"{key:15}| {line}", end="", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated agent keys: " + ",".join(agents.all_keys()))
    ap.add_argument("--solo", choices=agents.all_keys())
    ap.add_argument("--addresses", action="store_true")
    args = ap.parse_args()

    if args.solo:
        # Hide the "not enough funds for the Almanac contract" noise: the free Almanac API
        # registration is what Agentverse and ASI:One use, and it succeeds without funds.
        logging.getLogger("uagents.registration").setLevel(logging.ERROR)
        agents.build(args.solo).run()
        return

    keys = args.only.split(",") if args.only else agents.all_keys()
    unknown = set(keys) - set(agents.all_keys())
    if unknown:
        ap.error(f"unknown agents: {', '.join(sorted(unknown))}")

    if args.addresses:
        for k in agents.all_keys():
            print(f"{k:16} :{agents.port_for(k)}  {agents.address_for(k)}")
        return

    busy = [k for k in keys if port_in_use(agents.port_for(k))]
    if busy:
        sys.exit(f"Ports already in use for: {', '.join(busy)}. The agents are probably already running in "
                 "another terminal. Stop that one first (Ctrl+C there).")

    mode = "mailbox (Agentverse)" if config.MAILBOX else "local"
    print(f"Starting {len(keys)} Dermis agents · {mode}")
    for k in keys:
        print(f"  {k:16} :{agents.port_for(k)}  {agents.address_for(k)}")
    if config.MAILBOX:
        print("\nFirst run only: open each link, click Connect → Mailbox (leave this running):")
        for k in keys:
            print(f"  {k:16} https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A{agents.port_for(k)}"
                  f"&address={agents.address_for(k)}")
    print(flush=True)

    procs = []
    for k in keys:
        p = subprocess.Popen([sys.executable, __file__, "--solo", k], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        procs.append(p)
        threading.Thread(target=stream, args=(k, p), daemon=True).start()

    def stop(*_):
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=20)  # agents mark themselves inactive on Agentverse while stopping
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()
