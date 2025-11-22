// Decompress Unity .br build files to their expected names so the dev server serves them correctly.
// Usage: node scripts/decompress_build.js

const fs = require('fs');
const path = require('path');

const buildDir = path.join(__dirname, '..', 'public', 'Build');
const mapping = [
  { src: 'build.data.br', dest: 'build.data' },
  { src: 'build.framework.js.br', dest: 'build.framework.js' },
  { src: 'build.wasm.br', dest: 'build.wasm' },
];

async function decompress() {
  if (!fs.existsSync(buildDir)) {
    console.error('Build directory not found:', buildDir);
    process.exit(1);
  }

  for (const m of mapping) {
    const srcPath = path.join(buildDir, m.src);
    const destPath = path.join(buildDir, m.dest);
    if (!fs.existsSync(srcPath)) {
      console.warn('Source not found, skipping:', srcPath);
      continue;
    }
    try {
      const compressed = fs.readFileSync(srcPath);
      // Use built-in brotli decompression (Node 10.16+)
      const decompressed = require('zlib').brotliDecompressSync(compressed);
      fs.writeFileSync(destPath, decompressed);
      console.log(`Wrote ${destPath}`);
    } catch (err) {
      console.error('Failed to decompress', srcPath, err.message);
    }
  }
  console.log('Decompression complete.');
}

decompress();
