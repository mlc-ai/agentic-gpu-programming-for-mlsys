/* Local figures share selection, theme, and sizing behavior. */
(() => {
  function select(button) {
    const group = button.closest('[data-selection]');
    const key = button.dataset.select;
    group.querySelectorAll('[data-select]').forEach(item => {
      item.setAttribute('aria-pressed', String(item === button));
    });
    group.querySelectorAll('[data-panel]').forEach(panel => {
      panel.hidden = panel.dataset.panel !== key;
    });
    const highlights = (button.dataset.highlights || '').split(' ');
    group.querySelectorAll('[data-node]').forEach(node => {
      node.classList.toggle('active', highlights.includes(node.dataset.node));
    });
    if (button.dataset.state) group.dataset.state = button.dataset.state;
  }
  document.querySelectorAll('[data-select]').forEach(button => {
    button.addEventListener('click', () => select(button));
  });
  document.querySelectorAll('[data-select][aria-pressed="true"]').forEach(select);

  if (window.parent !== window) {
    const root = window.parent.document.documentElement;
    const syncTheme = () => {
      document.documentElement.dataset.theme = root.dataset.theme || 'light';
    };
    syncTheme();
    new MutationObserver(syncTheme).observe(root, {
      attributes: true, attributeFilter: ['data-theme']
    });
    const resize = () => {
      window.frameElement.style.height = `${Math.ceil(document.body.getBoundingClientRect().height) + 2}px`;
    };
    new ResizeObserver(resize).observe(document.body);
    resize();
  } else {
    const mode = window.matchMedia('(prefers-color-scheme: dark)');
    const syncTheme = () => { document.documentElement.dataset.theme = mode.matches ? 'dark' : 'light'; };
    syncTheme();
    mode.addEventListener('change', syncTheme);
  }
})();
