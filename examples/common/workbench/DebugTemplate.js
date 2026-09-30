export const debugTemplate = `<section class="debug-workspace" id="debugWorkspace" aria-labelledby="debugWorkspaceTitle" data-pane="analysis">
        <div class="debug-heading">
          <div class="debug-heading-copy"><div class="debug-title-line"><h2 id="debugWorkspaceTitle">Execution</h2><span id="debugState" class="debug-badge">IDLE</span></div><p id="debugSource">Current input · independent debug execution</p></div>
          <div class="debug-actions">
            <button id="debugOptionsToggle" class="secondary" aria-controls="debugOptions" aria-expanded="false">Capture settings</button>
            <button id="debugExport" class="secondary" disabled title="Download the execution plan, events, statistics and metadata">Export JSON</button>
          </div>
        </div>
        <div class="debug-toolbar" aria-label="Debug execution controls">
          <button id="debugStart" class="primary" disabled>Start session</button>
          <button id="debugContinue" class="primary" disabled title="Continue to the next breakpoint or finish (F8)">▶ Continue <kbd>F8</kbd></button>
          <button id="debugStep" class="secondary" disabled title="Execute one schedule entry (F10)">↦ Step <kbd>F10</kbd></button>
          <button id="debugCancel" class="secondary" disabled title="Cancel execution and preserve captured observations">■ Stop</button>
          <span class="debug-toolbar-divider"></span>
          <button id="debugCurrent" class="secondary" disabled title="Select and center the next execution step">Locate cursor</button>
          <button id="debugRunTo" class="secondary" disabled title="Continue to the selected step, respecting earlier breakpoints">Run to selected</button>
          <button id="debugRelease" class="secondary debug-toolbar-end" disabled title="Release the session and captured tensors">Release</button>
        </div>
        <div id="debugOptions" class="debug-settings" hidden>
          <label>Graph<select id="debugGraph"><option value="source">Original node boundaries</option><option value="optimized">Optimized graph</option></select></label>
          <label>Tensor capture<span class="debug-checkbox"><input type="checkbox" id="debugValues"> Retain raw values</span></label>
          <label class="debug-filter-field">Tensor names<input type="text" id="debugFilter" placeholder="All tensors, or exact names separated by commas"></label>
          <label>Capture budget<select id="debugBudget"><option value="16">16 MiB</option><option value="32">32 MiB</option><option value="64">64 MiB</option><option value="128">128 MiB</option></select></label>
          <p>Settings apply to the next session. Statistics are always captured. Filter tensors to conserve the capture budget; execution buffers and shared weights are separate.</p>
        </div>
        <div class="debug-status-line"><span id="debugStatus" role="status" aria-live="polite">Start a session to inspect the current input.</span><span id="debugCaptureMode">Statistics · original graph</span></div>
        <progress id="debugProgress" max="1" value="0" aria-label="Executed graph steps"></progress>
        <nav class="debug-pane-switch" aria-label="Workspace panes"><button class="secondary" data-debug-pane="execution">Execution</button><button class="secondary" data-debug-pane="analysis" aria-pressed="true">Analysis</button><button class="secondary" data-debug-pane="inspector">Inspector</button></nav>
        <div class="debug-main">
          <aside class="debug-outline" aria-label="Execution navigator">
            <div class="debug-pane-heading"><h3>Execution</h3><span id="debugNodeCount" class="debug-muted">No plan</span></div>
            <div class="debug-search"><input id="debugSearch" type="search" placeholder="Find node, op, tensor…" aria-label="Search execution nodes"><kbd>/</kbd></div>
            <div class="debug-outline-filters"><select id="debugNodeFilter" aria-label="Filter execution nodes"><option value="all">All nodes</option><option value="breakpoints">Breakpoints</option><option value="issues">Capture issues / nonfinite</option></select><button id="debugClearBreakpoints" class="secondary" disabled title="Clear all breakpoints">Clear breaks</button></div>
            <div id="debugNodes" class="debug-node-list" aria-label="Execution nodes"></div>
            <p id="debugNodesEmpty" class="debug-empty">Start a session to explore the graph.</p>
            <div class="debug-outline-footer"><span class="debug-break-dot" aria-hidden="true"></span><span>Click the gutter to set a breakpoint.<br>Stops <strong>before</strong> the selected node.</span></div>
          </aside>
          <div class="debug-resizer" data-resize="outline" role="separator" aria-label="Resize execution navigator" aria-orientation="vertical" tabindex="0"></div>

          <section class="debug-analysis" aria-label="Tensor and graph analysis">
            <div class="debug-inspector-tabs" id="debugAnalysisTabs" role="tablist" aria-label="Analysis views">
              <button id="debugGraphTab" role="tab" aria-selected="true" aria-controls="debugGraphPanel" data-debug-tab="graph">Graph</button>
              <button id="debugValuesTab" role="tab" aria-selected="false" aria-controls="debugValuesPanel" data-debug-tab="values" tabindex="-1">Tensor values</button>
            </div>
            <div id="debugAnalysisPanels" class="debug-analysis-panels">          <section class="debug-graph-pane" id="debugGraphPanel" role="tabpanel" aria-labelledby="debugGraphTab" aria-label="Interactive model graph">
            <div class="debug-pane-heading"><h3>Graph</h3><div class="debug-actions"><button id="debugGraphFocus" class="secondary" title="Center the selected node">Center</button><button id="debugGraphFit" class="secondary" title="Fit the entire graph">Fit</button></div></div>
            <div class="debug-graph-canvas" id="debugGraphCanvas"></div>
            <div class="debug-graph-empty" id="debugGraphEmpty"><span aria-hidden="true">◇</span><strong>Follow the data</strong><p>Start a session to inspect operator connections and captured tensors.</p></div>
            <div class="debug-graph-tools"><button id="debugZoomOut" class="secondary" aria-label="Zoom out">−</button><span id="debugGraphZoom">100%</span><button id="debugZoomIn" class="secondary" aria-label="Zoom in">+</button></div>
            <div class="debug-graph-legend"><span><i class="debug-legend-selected"></i>Selected</span><span><i class="debug-legend-next"></i>Next to run</span><span><i class="debug-legend-done"></i>Executed</span></div>
            <p class="debug-graph-help">Drag to pan · Ctrl/⌘ + scroll to zoom · click an edge to inspect its tensor</p>
          </section>
              <section id="debugValuesPanel" role="tabpanel" aria-labelledby="debugValuesTab" class="debug-tab-panel debug-analysis-scroll" hidden>
                <div id="debugValuesMissing" class="empty"><h3>No captured values yet</h3><p>Select an input or output in the tensor inspector. Capture just that tensor to keep memory bounded.</p><button id="debugCaptureSelected" class="primary" disabled>Capture selected tensor</button></div><section class="tensor-value-view" id="tensorValueView" aria-labelledby="tensorValueTitle">
                  <h3 id="tensorValueTitle">Tensor values</h3>
                  <p class="debug-note" id="tensorValueMessage" role="status" aria-live="polite">Select a tensor to browse its values.</p>
                  <div class="debug-recapture"><button id="tensorValueRecapture" hidden>Recapture this tensor</button><p id="tensorValueRecaptureNote" class="debug-note" hidden>Starts a new session and reruns the current input through the selected node. Only this tensor and input/output side are retained.</p></div>
                  <div class="debug-value-toolbar"><label>View<select id="tensorValueMode"><option value="table">Value table</option><option value="map">Page heatmap</option></select></label><label>Page size<select id="tensorValueCount" disabled><option value="64">64</option><option value="256" selected>256</option><option value="1024">1024</option></select></label><label class="debug-checkbox"><input id="tensorValueShade" type="checkbox" checked disabled> Shade table</label></div>
                  <details class="debug-index-lookup"><summary>Jump to offset / coordinates</summary><div class="debug-value-navigation"><label for="tensorValueOffset">Offset</label><input id="tensorValueOffset" type="text" inputmode="numeric" value="0" disabled><button id="tensorValueGo" class="secondary" disabled>Go</button><label for="tensorValueIndex">Index</label><input id="tensorValueIndex" type="text" placeholder="[0, 0, 0, 0]" disabled><button id="tensorValueJump" class="secondary" disabled>Jump</button></div></details>
                  <div class="debug-page-navigation"><button id="tensorValueFirst" class="secondary" disabled title="First page">«</button><button id="tensorValuePrevious" class="secondary" disabled title="Previous page">‹</button><span id="tensorValueRange" class="debug-note" aria-live="polite"></span><button id="tensorValueNext" class="secondary" disabled title="Next page">›</button><button id="tensorValueLast" class="secondary" disabled title="Last page">»</button></div>
                  <div class="debug-table-wrap" id="tensorValueTable"><table class="debug-table" aria-label="Tensor element values"><thead><tr><th>Index</th><th>Coordinates</th><th>Stored value</th><th>Dequantized</th></tr></thead><tbody id="tensorValueRows"></tbody></table></div>
                  <div id="tensorValueMapPanel" class="debug-value-map" hidden><canvas id="tensorValueMap" width="384" height="384" aria-label="Current page of stored tensor values as a heatmap"></canvas><p id="tensorValueMapReadout" class="debug-note">Hover a cell to inspect its value.</p><div class="debug-color-legend"><span id="tensorValueMapMin">—</span><i></i><span id="tensorValueMapMax">—</span></div><p class="debug-note">Flat page in row-major order; this is not a spatial slice. Magenta marks nonfinite values.</p></div>
                  <div id="tensorValueDistribution" class="debug-distribution" hidden><svg id="tensorValueHistogram" viewBox="0 0 320 60" role="img" aria-label="Distribution of stored values on the current page"></svg><p id="tensorValueHistogramCaption" class="debug-note"></p></div>
                  <div class="debug-actions"><button id="tensorValueCsv" class="secondary" disabled>Page CSV ↓</button><span class="debug-note">Only the displayed page is read.</span></div>
                </section>
              </section>
</div>
          </section>
          <div class="debug-resizer" data-resize="inspector" role="separator" aria-label="Resize tensor inspector" aria-orientation="vertical" tabindex="0"></div>
          <section class="debug-inspector" aria-label="Node and tensor inspector">
            <div class="debug-inspector-heading"><span class="debug-eyebrow">NODE INSPECTOR</span><h3 id="debugSelection">Select a node</h3><p id="debugNodeInfo">Inspect inputs, outputs and their captured values.</p></div>
            <div id="debugTensors" class="debug-tensor-list" aria-label="Node input and output tensors"></div>

            <div class="debug-inspector-tabs" role="tablist" aria-label="Tensor details"><button id="debugSummaryTab" role="tab" aria-selected="true" aria-controls="debugSummaryPanel" data-inspector-tab="summary">Summary</button><button id="debugMetadataTab" role="tab" aria-selected="false" aria-controls="debugMetadataPanel" data-inspector-tab="metadata" tabindex="-1">Metadata</button></div>
            <div class="debug-inspector-body">              <section id="debugSummaryPanel" role="tabpanel" aria-labelledby="debugSummaryTab" class="debug-tab-panel">
                <div id="debugTensorSummary" class="debug-tensor-summary"><div class="debug-empty">Choose an input or output tensor to inspect its shape, statistics and capture status.</div></div>
                <div class="debug-actions"><button id="debugShowValues" class="secondary" disabled>Explore values →</button></div>
              </section>
              <section id="debugMetadataPanel" role="tabpanel" aria-labelledby="debugMetadataTab" class="debug-tab-panel" hidden><p class="debug-note">Exact snapshot metadata for inspection and analysis tools.</p><pre id="debugPreview">Select a tensor.</pre></section>
            </div>
            <div class="debug-inspector-footer"><span id="debugSnapshotLabel" class="debug-muted">No tensor selected</span><button id="debugDownload" class="secondary" disabled>Raw tensor ↓</button></div>
          </section>
        </div>
        <div class="debug-footer"><div class="debug-capture-meter"><span class="debug-meter-track"><span id="debugMemoryBar"></span></span><span id="debugMemory">Capture store: —</span></div><span id="debugBreakCount">0 breakpoints</span><details class="debug-shortcuts"><summary title="Keyboard shortcuts">Shortcuts</summary><div><p><kbd>F8</kbd> Continue <kbd>F10</kbd> Step <kbd>F9</kbd> Breakpoint</p><p><kbd>/</kbd> Search <kbd>↑ ↓</kbd> Navigate list <kbd>Enter</kbd> Activate tab</p><a href="${new URL('../../../docs/debugging.md', import.meta.url)}" target="_blank" rel="noopener">Debugging guide ↗</a></div></details></div>
      </section>`;
