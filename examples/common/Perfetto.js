/** Open an engine trace using Perfetto's public embedding interface. */
const origin = 'https://ui.perfetto.dev';

function waitForPerfetto(popup) {
  return new Promise((resolve, reject) => {
    const finish = (error) => {
      clearInterval(pingTimer);
      clearTimeout(timeout);
      window.removeEventListener('message', onMessage);
      if (error) reject(error); else resolve();
    };
    const onMessage = (event) => {
      if (event.origin === origin && event.source === popup && event.data === 'PONG') finish();
    };
    const ping = () => {
      if (popup.closed) {
        finish(new Error('Perfetto closed before opening the trace. Keep the new tab open; the page must allow communication with it.'));
      } else {
        popup.postMessage('PING', origin);
      }
    };
    window.addEventListener('message', onMessage);
    const pingTimer = setInterval(ping, 200);
    const timeout = setTimeout(() => finish(new Error('Perfetto did not respond. Check your connection and try again.')), 30_000);
    ping();
  });
}

/** Call directly from a click handler so the browser can open the new tab. */
export async function openPerfetto(blob, fileName = 'volvoxai-trace.json') {
  if (!blob?.size) throw new Error('Capture or choose a trace first.');
  if (window.location.protocol === 'file:') throw new Error('Serve this page over HTTP to open Perfetto.');
  // Open synchronously while the click has user activation, then read metadata.
  const popup = window.open('about:blank', '_blank');
  if (!popup) throw new Error('Allow pop-ups for this page, then open Perfetto again.');
  try {
    const [buffer, {traceQueries}] = await Promise.all([
      blob.arrayBuffer(), import('./PerfettoQueries.js'),
    ]);
    const metadata = traceMetadata(buffer);
    const queries = traceQueries(metadata);
    // Surface known loss first. Otherwise start with useful data for this detail.
    const initialTitle = Number(metadata.droppedEvents) > 0 ? 'Capture'
      : metadata.detail === 'basic' ? 'Run summary' : 'Slow nodes';
    const initial = queries.findIndex(query => query.title === initialTitle);
    [queries[0], queries[initial]] = [queries[initial], queries[0]];
    // The last command selects the active table.
    const commands = queries.reverse().map(({title, sql}) => ({
      id: 'dev.perfetto.RunQueryAndShowTab', args: [sql, title],
    }));
    if (popup.closed) throw new Error('Perfetto closed before opening the trace.');
    popup.location = `${origin}/#!/?startupCommands=${encodeURIComponent(JSON.stringify(commands))}`;
    await waitForPerfetto(popup);
    if (popup.closed) throw new Error('Perfetto closed before opening the trace.');
    popup.postMessage({perfetto: {
      buffer, title: fileName, fileName, downloadable: true, shareable: false,
    }}, origin, [buffer]);
  } catch (error) {
    if (!popup.closed) popup.close();
    throw error;
  }
}

function traceMetadata(buffer) {
  const document = JSON.parse(new TextDecoder().decode(buffer));
  if (!Array.isArray(document?.traceEvents)) throw new Error('Choose a Chrome Trace JSON exported by VolvoxAI.');
  return document.otherData ?? {};
}
