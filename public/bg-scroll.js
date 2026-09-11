/* ============================================================
   Mounts the scrolling photo-wall background into a container.
   Auto-mounts into #bgScrollRoot on DOMContentLoaded if present.
   ============================================================ */

(function () {
  // Curated placeholder photography set (seeded so they're stable across loads).
  // Swap this array for real uploaded photo URLs whenever you're ready —
  // everything else (columns, looping, speeds) keeps working unchanged.
  const BG_IMAGES = [
    'https://picsum.photos/seed/wsstudio01/420/620',
    'https://picsum.photos/seed/wsstudio02/420/620',
    'https://picsum.photos/seed/wsstudio03/420/620',
    'https://picsum.photos/seed/wsstudio04/420/620',
    'https://picsum.photos/seed/wsstudio05/420/620',
    'https://picsum.photos/seed/wsstudio06/420/620',
    'https://picsum.photos/seed/wsstudio07/420/620',
    'https://picsum.photos/seed/wsstudio08/420/620',
    'https://picsum.photos/seed/wsstudio09/420/620',
    'https://picsum.photos/seed/wsstudio10/420/620',
    'https://picsum.photos/seed/wsstudio11/420/620',
    'https://picsum.photos/seed/wsstudio12/420/620'
  ];

  const COLUMN_COUNT = 6;
  const COLUMN_DURATIONS = [42, 55, 38, 60, 46, 50]; // seconds — varied so columns don't sync visually
  const COLUMN_DELAYS = [0, -12, -5, -30, -18, -8]; // negative delay = starts mid-cycle, avoids a "just started" look

  function buildColumn(images, index) {
    const col = document.createElement('div');
    col.className = 'bg-scroll-col';
    col.style.animationDuration = `${COLUMN_DURATIONS[index % COLUMN_DURATIONS.length]}s`;
    col.style.animationDelay = `${COLUMN_DELAYS[index % COLUMN_DELAYS.length]}s`;

    // Render the list twice back-to-back so translateY(-50%) loops seamlessly.
    [images, images].forEach(set => {
      set.forEach(src => {
        const img = document.createElement('img');
        img.src = src;
        img.alt = '';
        img.loading = 'lazy';
        col.appendChild(img);
      });
    });
    return col;
  }

  function mount(containerId) {
    const root = document.getElementById(containerId || 'bgScrollRoot');
    if (!root || root.dataset.mounted) return;
    root.dataset.mounted = 'true';
    root.classList.add('bg-scroll-wrap');

    const columnsEl = document.createElement('div');
    columnsEl.className = 'bg-scroll-columns';

    for (let i = 0; i < COLUMN_COUNT; i++) {
      // Round-robin a shuffled-feeling slice of images into each column
      const colImages = [];
      for (let j = 0; j < 4; j++) {
        colImages.push(BG_IMAGES[(i * 3 + j * 2) % BG_IMAGES.length]);
      }
      columnsEl.appendChild(buildColumn(colImages, i));
    }

    root.appendChild(columnsEl);

    const overlay = document.createElement('div');
    overlay.className = 'bg-scroll-overlay';
    document.body.appendChild(overlay);
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('bgScrollRoot')) mount('bgScrollRoot');
  });

  window.BGScroll = { mount };
})();
