/** Presentation helpers. Engine operations stay behind the generated API. */
export function mountView(root, template, styles = []) {
  const view = root.attachShadow({mode: 'open'});
  for (const file of ['Base.css', ...styles]) {
    const link = document.createElement('link'); link.rel = 'stylesheet'; link.href = new URL(file, import.meta.url); view.append(link);
  }
  const content = document.createElement('template'); content.innerHTML = template; view.append(content.content.cloneNode(true));
  return view;
}

export function element(tag, className = '', text) {
  const value = document.createElement(tag); value.className = className;
  if (text !== undefined) value.textContent = String(text);
  return value;
}

export function download(blob, filename) {
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function jsonBlob(value) {
  return new Blob([JSON.stringify(value, (_, v) => typeof v === 'bigint' ? String(v) : v, 2)], {type: 'application/json'});
}

export function formatBytes(value) {
  if (value === undefined || value === null) return '—';
  const bytes = Number(value);
  return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(2)} MiB` : bytes >= 1024 ? `${(bytes / 1024).toFixed(1)} KiB` : `${bytes} B`;
}
export const formatMs = value => value === null || value === undefined ? '—' : `${value.toFixed(2)} ms`;

/** Manual activation: arrow keys never trigger raw tensor reads or execution. */
export function tabNavigation(buttons) {
  for (const button of buttons) button.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const tabs = [...buttons].filter(tab => !tab.hidden && !tab.disabled), index = tabs.indexOf(button);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 :
      (index + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length;
    tabs[next]?.focus();
  });
}
