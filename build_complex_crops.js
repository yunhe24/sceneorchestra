const fs = require('fs');
const path = require('path');

const assets = path.join(__dirname, 'assets');

function cropSvg(source, output, raster, crop) {
  const sourcePath = path.join(assets, source);
  let svg = fs.readFileSync(sourcePath, 'utf8');
  const match = svg.match(/<svg[^>]*width="([0-9.]+)pt"[^>]*height="([0-9.]+)pt"[^>]*viewBox="0 0 ([0-9.]+) ([0-9.]+)"/);
  if (!match) throw new Error(`Cannot parse SVG geometry: ${source}`);
  const pageW = Number(match[3]);
  const pageH = Number(match[4]);
  const [rw, rh] = raster;
  const [cx, cy, cw, ch] = crop;
  const x = cx / rw * pageW;
  const y = cy / rh * pageH;
  const w = cw / rw * pageW;
  const h = ch / rh * pageH;
  svg = svg.replace(
    /width="[0-9.]+pt" height="[0-9.]+pt" viewBox="0 0 [0-9.]+ [0-9.]+"/,
    `width="${w.toFixed(4)}pt" height="${h.toFixed(4)}pt" viewBox="${x.toFixed(4)} ${y.toFixed(4)} ${w.toFixed(4)} ${h.toFixed(4)}"`,
  );
  fs.writeFileSync(path.join(assets, output), svg);
}

const main = [3600, 2495];
const supp = [4200, 4498];
const crops = {
  'complex_meeting_sw.svg': ['complex.svg', main, [520, 840, 1020, 780]],
  'complex_meeting_ours.svg': ['complex.svg', main, [520, 1680, 1020, 780]],
  'complex_garage_sw.svg': ['complex.svg', main, [1600, 840, 1020, 780]],
  'complex_garage_ours.svg': ['complex.svg', main, [1600, 1680, 1020, 780]],
  'complex_laundry_sw.svg': ['complex.svg', main, [2720, 840, 820, 780]],
  'complex_laundry_ours.svg': ['complex.svg', main, [2720, 1680, 820, 780]],
  'complex_bathroom_sw.svg': ['complex_supp_full.svg', supp, [430, 610, 810, 700]],
  'complex_bathroom_ours.svg': ['complex_supp_full.svg', supp, [430, 1460, 810, 690]],
  'complex_office_sw.svg': ['complex_supp_full.svg', supp, [1730, 610, 810, 700]],
  'complex_office_ours.svg': ['complex_supp_full.svg', supp, [1730, 1460, 810, 690]],
  'complex_nursery_sw.svg': ['complex_supp_full.svg', supp, [3030, 610, 820, 700]],
  'complex_nursery_ours.svg': ['complex_supp_full.svg', supp, [3030, 1460, 820, 690]],
  'complex_living_sw.svg': ['complex_supp_full.svg', supp, [430, 2950, 820, 700]],
  'complex_living_ours.svg': ['complex_supp_full.svg', supp, [350, 3780, 930, 700]],
  'complex_game_sw.svg': ['complex_supp_full.svg', supp, [1430, 2950, 760, 700]],
  'complex_game_ours.svg': ['complex_supp_full.svg', supp, [1430, 3780, 760, 700]],
  'complex_medical_sw.svg': ['complex_supp_full.svg', supp, [2320, 2950, 800, 700]],
  'complex_medical_ours.svg': ['complex_supp_full.svg', supp, [2320, 3780, 800, 700]],
  'complex_studio_sw.svg': ['complex_supp_full.svg', supp, [3210, 2950, 950, 700]],
  'complex_studio_ours.svg': ['complex_supp_full.svg', supp, [3260, 3780, 850, 700]],
};

for (const [output, [source, raster, crop]] of Object.entries(crops)) {
  cropSvg(source, output, raster, crop);
}

console.log(`Generated ${Object.keys(crops).length} complex-scene crops.`);
