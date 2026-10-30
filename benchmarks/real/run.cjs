// Reproducible UCI workload runner for the MoonBit JS backend.
// Usage: node benchmarks/real/run.cjs pendigits|letters|winequality|optdigits
// The official source files are cached under the OS temp directory, outside
// the repository. No generated data or third-party JS dependencies are used.
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const cp = require('node:child_process');

const root = path.resolve(__dirname, '..', '..');
const cache = path.join(os.tmpdir(), 'moonhnsw-real-bench');
const sources = {
  pendigits_train: 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra',
  pendigits_test: 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes',
  letters: 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data',
  winequality: 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
  optdigits_train: 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
  optdigits_test: 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',
};
const expectedSha256 = {
  pendigits_train: 'e2b9eb9f0d0467e2b64a4816a3420edf2b8043447576f4b84337aba44a9f97d3',
  pendigits_test: '8bd03229c5c5291fefe43e45465dd948d2645bf23328b9d993e0b777666b2015',
  letters: '2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2',
  winequality: '76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836',
  optdigits_train: 'e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd',
  optdigits_test: '6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8',
};

async function source(name) {
  await fsp.mkdir(cache, { recursive: true });
  const file = path.join(cache, name + '.csv');
  if (!fs.existsSync(file)) {
    const response = await fetch(sources[name]);
    if (!response.ok) throw new Error(`download ${sources[name]}: HTTP ${response.status}`);
    await fsp.writeFile(file, Buffer.from(await response.arrayBuffer()));
  }
  const bytes = await fsp.readFile(file);
  const sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
  if (sha256 !== expectedSha256[name]) {
    throw new Error(`official source hash changed for ${name}: ${sha256}`);
  }
  console.log(`source=${sources[name]} sha256=${sha256}`);
  return bytes.toString('utf8').trim().split(/\r?\n/);
}

function flatten(rows, first, last, labelFirst, dim = 16) {
  const out = new Float32Array((last - first) * dim);
  for (let i = first; i < last; i++) {
    const cells = rows[i].split(',');
    if (cells.length !== dim + 1) throw new Error(`bad CSV row ${i + 1}: ${cells.length} fields`);
    const offset = labelFirst ? 1 : 0;
    for (let j = 0; j < dim; j++) {
      const raw = cells[offset + j].trim();
      if (raw === '') throw new Error(`missing value at row ${i + 1}, col ${j + 1}`);
      const value = Number(raw);
      if (!Number.isFinite(value)) throw new Error(`non-finite value at row ${i + 1}, col ${j + 1}`);
      out[(i - first) * dim + j] = value;
    }
  }
  return out;
}

function flattenWine(rows, first, last) {
  const out = new Float32Array((last - first) * 11);
  for (let i = first; i < last; i++) {
    const cells = rows[i].split(';');
    if (cells.length !== 12) throw new Error(`bad wine CSV row ${i + 1}: ${cells.length} fields`);
    for (let j = 0; j < 11; j++) {
      const raw = cells[j].trim();
      if (raw === '') throw new Error(`missing wine value at row ${i + 1}, col ${j + 1}`);
      const value = Number(raw);
      if (!Number.isFinite(value)) throw new Error(`non-finite wine value at row ${i + 1}, col ${j + 1}`);
      out[(i - first) * 11 + j] = value;
    }
  }
  return out;
}

async function main() {
  const selected = process.argv[2];
  if (!['pendigits', 'letters', 'winequality', 'optdigits'].includes(selected)) {
    throw new Error('usage: node benchmarks/real/run.cjs pendigits|letters|winequality|optdigits');
  }
  let train, test, dim;
  if (selected === 'pendigits') {
    const trainRows = await source('pendigits_train');
    const testRows = await source('pendigits_test');
    if (trainRows.length !== 7494 || testRows.length !== 3498) {
      throw new Error(`unexpected pendigits split: ${trainRows.length}/${testRows.length}`);
    }
    train = flatten(trainRows, 0, trainRows.length, false);
    test = flatten(testRows, 0, 200, false);
    dim = 16;
  } else if (selected === 'letters') {
    const rows = await source('letters');
    if (rows.length !== 20000) throw new Error(`unexpected letters size: ${rows.length}`);
    train = flatten(rows, 0, 16000, true);
    test = flatten(rows, 16000, 16200, true);
    dim = 16;
  } else if (selected === 'winequality') {
    const rows = await source('winequality');
    if (rows.length !== 4899) throw new Error(`unexpected wine quality size: ${rows.length}`);
    train = flattenWine(rows, 1, 4001);
    test = flattenWine(rows, 4001, 4201);
    dim = 11;
  } else {
    const trainRows = await source('optdigits_train');
    const testRows = await source('optdigits_test');
    if (trainRows.length !== 3823 || testRows.length !== 1797) {
      throw new Error(`unexpected optdigits split: ${trainRows.length}/${testRows.length}`);
    }
    train = flatten(trainRows, 0, trainRows.length, false, 64);
    test = flatten(testRows, 0, 200, false, 64);
    dim = 64;
  }
  globalThis.__moonhnsw_dataset = { name: selected, dim, train, test };
  const moduleVersion = fs.readFileSync(path.join(root, 'moon.mod'), 'utf8').match(/^version\s*=\s*"([^"]+)"/m)?.[1];
  if (!moduleVersion) throw new Error('moon.mod version not found');
  console.log(`module=yiwnagmolu/moonhnsw version=${moduleVersion}`);
  console.log(`runtime=${process.version} platform=${process.platform}/${process.arch} cpu=${os.cpus()[0].model.trim()}`);
  const fallback = path.join(os.homedir(), '.moon', 'bin', process.platform === 'win32' ? 'moon.exe' : 'moon');
  const moon = process.env.MOON_BIN || (fs.existsSync(fallback) ? fallback : 'moon');
  cp.execFileSync(moon, ['build', '--target', 'js', '--release', 'benchmarks/real'], {
    cwd: root,
    stdio: 'inherit',
  });
  require(path.join(root, '_build', 'js', 'release', 'build', 'benchmarks', 'real', 'real.js'));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
