const tabs = [...document.querySelectorAll('.tab')];
const panels = [...document.querySelectorAll('.tab-panel')];
tabs.forEach(tab => tab.addEventListener('click', () => {
  tabs.forEach(t => { const on = t === tab; t.classList.toggle('active', on); t.setAttribute('aria-selected', String(on)); });
  panels.forEach(panel => { const on = panel.id === tab.dataset.panel; panel.hidden = !on; panel.classList.toggle('active', on); });
}));

const lightbox = document.getElementById('lightbox');
const lightboxImage = lightbox.querySelector('img');
const closeLightbox = () => { lightbox.classList.remove('open'); lightbox.setAttribute('aria-hidden', 'true'); lightboxImage.src = ''; };
document.querySelectorAll('.zoomable').forEach(image => image.addEventListener('click', () => {
  lightboxImage.src = image.currentSrc || image.src;
  lightboxImage.alt = image.alt;
  lightbox.classList.add('open');
  lightbox.setAttribute('aria-hidden', 'false');
}));
lightbox.addEventListener('click', event => { if (event.target === lightbox || event.target.classList.contains('lightbox-close')) closeLightbox(); });
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeLightbox(); });

const citation = document.getElementById('citationText').innerText;
const copyButton = document.getElementById('copyCitation');
const copyStatus = document.querySelector('.copy-status');
copyButton.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(citation); copyStatus.textContent = 'Copied'; }
  catch { copyStatus.textContent = 'Select and copy from the block'; }
  window.setTimeout(() => { copyStatus.textContent = ''; }, 2200);
});
