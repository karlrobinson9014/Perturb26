// pm2 apps for Perturb26.
//   pm2 start harness/ecosystem.config.js --only perturb26-loop    # the autonomous loop
//   pm2 start harness/ecosystem.config.js --only perturb26-miner   # live miner (needs miner.env)
// Do not run both on one GPU: the loop's benchmarks and the live miner would
// fight for it and both would get worse numbers.
const path = require("path");
const root = path.resolve(__dirname, "..");

module.exports = {
  apps: [
    {
      name: "perturb26-loop",
      script: path.join(root, "harness", "loop.sh"),
      interpreter: "bash",
      cwd: root,
      autorestart: true,
      stop_exit_codes: [0], // STOP / MAX_ITERS / target reached exit 0 and stay stopped
      restart_delay: 60000,
      max_restarts: 20,
      kill_timeout: 60000,
      time: true,
      out_file: path.join(root, "logs", "pm2-loop.out.log"),
      error_file: path.join(root, "logs", "pm2-loop.err.log"),
    },
    {
      name: "perturb26-dashboard",
      script: path.join(root, "harness", "dashboard.py"),
      interpreter: path.join(root, ".venv", "bin", "python"),
      args: "--port 18080",
      cwd: root,
      autorestart: true,
      restart_delay: 5000,
      time: true,
      out_file: path.join(root, "logs", "pm2-dashboard.out.log"),
      error_file: path.join(root, "logs", "pm2-dashboard.err.log"),
    },
    {
      name: "perturb26-miner",
      script: path.join(root, "harness", "run_miner.sh"),
      interpreter: "bash",
      cwd: root,
      autorestart: true,
      restart_delay: 10000,
      time: true,
      out_file: path.join(root, "logs", "pm2-miner.out.log"),
      error_file: path.join(root, "logs", "pm2-miner.err.log"),
    },
  ],
};
