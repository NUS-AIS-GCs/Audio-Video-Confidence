// The shared ThreeHeadResults component is rendered on the authenticated server.
// Build inert DOM instructions without interpreting HTML or model text.
const AVLiveResearch = {
  render(element, result) {
    const tags = new Set(['div','section','span','h4','h5','p','ul','li','strong','b',
      'meter','details','summary','pre','table','thead','tbody','tr','th','td','br']);
    const attrs = new Set(['class','role','aria-live','aria-label','title','data-head',
      'data-state','data-av-raw-result','min','max','value','scope']);
    try {
      if (!result || !Number.isFinite(result.score) || !Number.isFinite(result.threshold) || !result.result_view)
        throw new Error('INVALID_RESULT_VIEW');
      function build(view, depth = 0) {
        if (depth > 20) throw new Error('INVALID_RESULT_VIEW');
        if (typeof view === 'string') {
          return element.ownerDocument.createTextNode(view);
        }
        if (!tags.has(view.tag) || !Array.isArray(view.children)) throw new Error('INVALID_RESULT_VIEW');
        const node = element.ownerDocument.createElement(view.tag);
        for (const [key, value] of Object.entries(view.attrs)) {
          if (!attrs.has(key) || typeof value !== 'string') throw new Error('INVALID_RESULT_VIEW');
          node.setAttribute(key, value);
        }
        for (const child of view.children) node.appendChild(build(child, depth + 1));
        return node;
      }
      element.replaceChildren(build(result.result_view));
    } catch (_) {
      element.textContent = 'UNABLE TO VERIFY. The validated result could not be displayed. Reset and try again.';
    }
  }
};
if (typeof module !== 'undefined' && module.exports) module.exports = AVLiveResearch;
