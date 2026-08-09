export function childProcessHasSettled(child) {
  return child.exitCode != null || child.signalCode != null;
}

export function observeChildProcessExit(child) {
  if (childProcessHasSettled(child)) return Promise.resolve();
  return new Promise((resolve) => {
    const settle = () => {
      child.removeListener('exit', settle);
      child.removeListener('error', settle);
      resolve();
    };
    child.once('exit', settle);
    // A spawn failure does not necessarily emit `exit`, but it is equally
    // terminal for the benchmark supervisor.
    child.once('error', settle);
  });
}
