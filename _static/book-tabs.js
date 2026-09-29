/* Launcher tabs keep normal book markup, highlighting, and copy buttons. */
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.launch-tabs').forEach(group => {
    const list = group.querySelector('.launch-tablist');
    const tabs = [...list.querySelectorAll('button')];
    const panels = tabs.map(tab => document.getElementById(tab.getAttribute('aria-controls')));
    const select = index => {
      tabs.forEach((tab, i) => {
        tab.setAttribute('aria-selected', String(i === index));
        tab.tabIndex = i === index ? 0 : -1;
        panels[i].hidden = i !== index;
      });
    };
    const selectAnchor = () => {
      const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      const index = panels.findIndex(panel => panel === target || panel.contains(target));
      if (index >= 0) select(index);
    };
    list.hidden = false;
    list.setAttribute('role', 'tablist');
    tabs.forEach((tab, i) => {
      tab.setAttribute('role', 'tab');
      panels[i].setAttribute('role', 'tabpanel');
      panels[i].setAttribute('aria-labelledby', tab.id);
      tab.addEventListener('click', () => select(i));
      tab.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight') next = (i + 1) % tabs.length;
        else if (event.key === 'ArrowLeft') next = (i + tabs.length - 1) % tabs.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = tabs.length - 1;
        else return;
        event.preventDefault();
        select(next);
        tabs[next].focus();
      });
    });
    select(0);
    selectAnchor();
    window.addEventListener('hashchange', selectAnchor);
  });
});
