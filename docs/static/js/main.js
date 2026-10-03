// Theme toggle (persists per viewer; falls back to the OS setting)
(function () {
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem('beda-theme');
    if (saved === 'dark' || saved === 'light') root.setAttribute('data-theme', saved);
  } catch (e) {}

  var btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.addEventListener('click', function () {
      var dark = root.getAttribute('data-theme') === 'dark' ||
        (!root.hasAttribute('data-theme') &&
         window.matchMedia('(prefers-color-scheme: dark)').matches);
      var next = dark ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('beda-theme', next); } catch (e) {}
    });
  }
})();

// Play looping demo clips only while they are on screen
(function () {
  var clips = document.querySelectorAll('.vcard video[loop]');
  if (!clips.length || !('IntersectionObserver' in window)) return;
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      var v = en.target;
      if (en.isIntersecting) { v.play().catch(function () {}); }
      else { v.pause(); }
    });
  }, { threshold: 0.25 });
  clips.forEach(function (v) { io.observe(v); });
})();
