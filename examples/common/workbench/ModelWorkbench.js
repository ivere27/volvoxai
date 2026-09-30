import {mountView, tabNavigation} from './View.js';
import {ModelDebugger} from './ModelDebugger.js';
import {ProfileView} from './ProfileView.js';

/** Reusable application shell. All model policy is supplied by the embedding example. */
export class ModelWorkbench {
  #root; #e;
  constructor({root, pb, title, subtitle, debug, onRecord, onBusyChange = () => {}}) {
    this.#root = mountView(root, `
      <main class="workbench">
        <header class="workbench-heading">
          <div class="workbench-identity"><div class="brand-mark" aria-hidden="true">Vx</div><h1 id="workbenchTitle"></h1></div>
          <nav class="workbench-tabs tabs" role="tablist" aria-label="Model workflows"><button id="workbenchInputTab" role="tab" aria-controls="workbenchInput" aria-selected="true" data-workbench-tab="input">Input &amp; results</button><button id="workbenchDebugTab" role="tab" aria-controls="workbenchDebug" aria-selected="false" data-workbench-tab="debug" tabindex="-1">Debug</button><button id="workbenchProfileTab" role="tab" aria-controls="workbenchProfile" aria-selected="false" data-workbench-tab="profile" tabindex="-1">Performance &amp; memory</button></nav>
          <div id="workbenchControls" class="row"></div>
          <div class="workbench-status"><span id="workbenchStatus" class="badge" role="status" aria-live="polite">Preparing model…</span><a class="workbench-help" href="${new URL('../../../docs/debugging.md', import.meta.url)}" target="_blank" rel="noopener" aria-label="Workbench guide">Guide ↗</a></div>
        </header>
        <p id="workbenchError" class="notice" role="alert" hidden></p>
        <section id="workbenchInput" class="workbench-input" role="tabpanel" aria-labelledby="workbenchInputTab"></section>
        <section id="workbenchDebug" class="workbench-tool" role="tabpanel" aria-labelledby="workbenchDebugTab" hidden><div id="workbenchDebugger"></div></section>
        <section id="workbenchProfile" class="workbench-tool" role="tabpanel" aria-labelledby="workbenchProfileTab" hidden><div id="workbenchProfiler"></div></section>
        <footer class="workbench-footer"><span>Unified edge engine · Native C / GPU · WASM / WebGPU</span><span>Local model execution · explicit debug / profile capture</span></footer>
      </main>`, ['Workbench.css']);
    this.#e = Object.fromEntries(['Title', 'Status', 'Error', 'Controls', 'Input'].map(k => [k, this.#root.getElementById(`workbench${k}`)]));
    this.#e.Title.textContent = title; this.#e.Title.title = subtitle;
    const tabs = this.#root.querySelectorAll('[data-workbench-tab]');
    for (const button of tabs) button.onclick = () => this.show(button.dataset.workbenchTab);
    tabNavigation(tabs);
    this.debugger = new ModelDebugger({root: this.#root.getElementById('workbenchDebugger'), pb, ...debug, onBusyChange});
    this.profiler = new ProfileView({root: this.#root.getElementById('workbenchProfiler'), pb, onRecord,
      onInspect: async name => { this.show('debug'); await this.debugger.inspectTensor(name); }});
  }

  /** The embedding model mounts its own inputs/results and model selectors here. */
  get inputRoot() { return this.#e.Input; }
  get controlsRoot() { return this.#e.Controls; }

  show(name) {
    for (const button of this.#root.querySelectorAll('[data-workbench-tab]')) {
      const active = button.dataset.workbenchTab === name;
      button.setAttribute('aria-selected', String(active)); button.tabIndex = active ? 0 : -1;
      this.#root.getElementById(button.getAttribute('aria-controls')).hidden = !active;
    }
  }
  setStatus(text, {error = false} = {}) {
    this.#e.Status.textContent = error ? 'Error' : text; this.#e.Status.title = text;
    this.#e.Status.dataset.error = String(error); this.#e.Error.hidden = !error; this.#e.Error.textContent = error ? text : '';
  }
  async dispose() { await this.debugger.dispose(); this.profiler.reset(); this.#root.replaceChildren(); }
}
