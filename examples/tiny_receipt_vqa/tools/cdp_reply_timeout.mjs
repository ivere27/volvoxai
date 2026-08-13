const CONTROL_REPLY_TIMEOUT_MS = 5000;

export function cdpReplyTimeoutMs(method, remainingMs) {
  return method === 'Runtime.evaluate'
    ? remainingMs
    : Math.min(remainingMs, CONTROL_REPLY_TIMEOUT_MS);
}
