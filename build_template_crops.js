const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const assets = path.join(__dirname, 'assets');
const converter = '/opt/homebrew/bin/rsvg-convert';

function renderCrop(source, output, raster, crop, width = 900) {
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
  const temporary = path.join(os.tmpdir(), `sceneorchestra-${output.replace('.png', '.svg')}`);
  fs.writeFileSync(temporary, svg);
  const result = spawnSync(converter, ['-w', String(width), temporary, '-o', path.join(assets, output)], { stdio: 'inherit' });
  fs.unlinkSync(temporary);
  if (result.status !== 0) throw new Error(`Failed to render ${output}`);
}

const mainRaster = [5600, 3492];
const mainColumns = {
  layoutgpt: [520, 1000],
  holodeck: [1530, 1000],
  idesign: [2600, 1000],
  sceneweaver: [3650, 1000],
  ours: [4610, 990],
};
const mainRows = {
  bathroom: [200, 750],
  meeting: [980, 780],
  bedroom: [1800, 830],
  gym: [2670, 822],
};
for (const [room, [y, h]] of Object.entries(mainRows)) {
  for (const [method, [x, w]] of Object.entries(mainColumns)) {
    renderCrop('qual.svg', `template_${room}_${method}.png`, mainRaster, [x, y, w, h]);
  }
}

// The supplementary figure uses images with different aspect ratios and
// placements. Exact per-image bounds avoid clipping scenes or leaking labels
// and neighbouring results into the crop.
const suppRaster = [2072.27, 2600];
const suppImages = {
  living: {
    layoutgpt: [260.0284, 54.481578, 908 * 0.604809, 646 * 0.603874 - 8],
    holodeck: [272.983836, 456.737057, 799 * 0.583724, 669 * 0.584189],
    idesign: [298.894708, 877.068986, 572 * 0.64299, 661 * 0.643525],
    sceneweaver: [267.308653, 1331.228999, 964 * 0.569675 - 8, 686 * 0.569712],
    ours1: [233.397781, 1765.955857, 980 * 0.613253, 646 * 0.612787],
    ours2: [233.397781, 2197.083984, 971 * 0.57891, 684 * 0.578743],
  },
  restaurant: {
    layoutgpt: [833.666324, 55.118535, 838 * 0.735206, 519 * 0.735001 - 3],
    holodeck: [819.271395, 427.947199, 986 * 0.65332, 643 * 0.653704],
    idesign: [696.194751, 930.330224, 764 * 1.055126, 303 * 1.054678],
    sceneweaver: [881.16959, 1333.388238, 721 * 0.721743, 540 * 0.722412],
    ours1: [885.488068, 1765.955857, 727 * 0.658365, 601 * 0.65867],
    ours2: [820.710888, 2197.083984, 1101 * 0.535397, 739 * 0.535671],
  },
  waiting: {
    layoutgpt: [1474.240668, 55.118535, 834 * 0.685226, 544 * 0.685347],
    holodeck: [1517.425455, 442.342128, 813 * 0.595805, 680 * 0.595908],
    idesign: [1544.056074, 884.266451, 620 * 0.695368, 590 * 0.695348],
    sceneweaver: [1550.533792, 1331.228999, 756 * 0.553138, 707 * 0.55279],
    ours1: [1415.221459, 1765.955857, 985 * 0.659828, 600 * 0.659768],
    ours2: [1434.654613, 2197.083984, 989 * 0.637511, 613 * 0.637557],
  },
};
for (const [room, methods] of Object.entries(suppImages)) {
  for (const [method, crop] of Object.entries(methods)) {
    renderCrop('qual_supp.svg', `template_${room}_${method}.png`, suppRaster, crop);
  }
}

console.log('Generated 38 template-instruction crops.');
