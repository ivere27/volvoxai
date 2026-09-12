/** Engine conformance fixtures inspect domain refusals as protobuf responses.
 * Product callers use EngineHost directly and receive rejected Promises. */
import { reportError } from '../ts/host/OperationReports.js';

export function checkedReport(response, operation = 'fixture') {
  const report = 'report' in response ? response.report : response;
  const error = reportError(response, report, operation);
  if (error) throw error;
  return response;
}

export function reportTransport(transport) {
  const wrap = call => ({
    send: data => call.send(data), halfClose: () => call.halfClose(),
    cancel: code => call.cancel(code), close: () => call.close(),
    async recv() {
      try { return await call.recv(); } catch (error) {
        // EngineHost delivers the complete protobuf before interpreting its
        // report on terminal receive. Transport failures remain failures.
        if (error?.name === 'VolvoxAIError' && error.response !== null) return null;
        throw error;
      }
    },
  });
  return {
    async open(method, options) { return wrap(await transport.open(method, options)); },
    async openWithRequest(method, encode, options) {
      if (transport.openWithRequest) return wrap(await transport.openWithRequest(method, encode, options));
      const data = encode();
      const call = await transport.open(method, options);
      try { await call.send(data); await call.halfClose(); return wrap(call); }
      catch (error) { await call.close(); throw error; }
    },
  };
}
