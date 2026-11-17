// Pack the built Firefox extension into a ZIP with POSIX-style entry names.
//
// This replaces the earlier PowerShell Compress-Archive flow (which wrote
// backslash (`\`) separators on Windows and caused AMO validation to fail
// with "Invalid file name in archive: assets\App-..."). Every entry here
// uses forward slashes.
//
// Usage: node scripts/pack-firefox.mjs
//   Source dir (default): dist/firefox
//   Output (default): <Desktop>/PageGlance-Releases/PageGlance-Firefox-v0.1.0.zip
//   Override via PACK_SRC_DIR and PACK_OUT_ZIP env vars.

import { deflateRawSync } from 'node:zlib';
import {
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
  renameSync,
  rmSync,
  mkdirSync,
  existsSync,
} from 'node:fs';
import { join, relative, sep, dirname } from 'node:path';
import { homedir } from 'node:os';

const SRC_DIR = process.env.PACK_SRC_DIR || 'dist/firefox';
const OUT_ZIP =
  process.env.PACK_OUT_ZIP ||
  join(
    homedir(),
    'Desktop',
    'PageGlance-Releases',
    'PageGlance-Firefox-v0.1.1.zip',
  );

// ---- utility: POSIX relative path with forward slashes ----
function posixRel(root, abs) {
  return relative(root, abs).split(sep).join('/');
}

// ---- CRC-32 ----
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

// ---- DOS date/time encoding ----
function dosDateTime(date) {
  const time =
    (date.getHours() << 11) |
    (date.getMinutes() << 5) |
    (date.getSeconds() >> 1);
  const dosDate =
    ((date.getFullYear() - 1980) << 9) |
    ((date.getMonth() + 1) << 5) |
    date.getDate();
  return { time, date: dosDate };
}

// ---- collect files ----
function collectFiles(dir) {
  const out = [];
  const walk = (cur) => {
    for (const name of readdirSync(cur)) {
      const full = join(cur, name);
      const st = statSync(full);
      if (st.isDirectory()) walk(full);
      else out.push(full);
    }
  };
  walk(dir);
  return out.sort();
}

// ---- build zip bytes ----
function buildZip(root, files) {
  const entries = [];
  const chunks = [];
  let localOffset = 0;

  for (const abs of files) {
    const name = posixRel(root, abs);
    const data = readFileSync(abs);
    const crc = crc32(data);
    const { time, date } = dosDateTime(statSync(abs).mtime);

    const deflated = deflateRawSync(data, { level: 9 });
    const comp = deflated.length < data.length ? deflated : data;
    const method = deflated.length < data.length ? 8 : 0;

    const nameBuf = Buffer.from(name, 'utf8');
    const flags = 0x0800; // UTF-8 file names

    // Local file header
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); // signature
    local.writeUInt16LE(20, 4); // version needed
    local.writeUInt16LE(flags, 6);
    local.writeUInt16LE(method, 8);
    local.writeUInt16LE(time, 10);
    local.writeUInt16LE(date, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(comp.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nameBuf.length, 26);
    local.writeUInt16LE(0, 28); // extra len

    chunks.push(local, nameBuf, comp);

    entries.push({
      nameBuf,
      method,
      flags,
      time,
      date,
      crc,
      compSize: comp.length,
      uncompSize: data.length,
      localOffset,
    });

    localOffset += local.length + nameBuf.length + comp.length;
  }

  // Central directory
  const centralStart = chunks.reduce((n, c) => n + c.length, 0);
  const central = [];
  for (const e of entries) {
    const c = Buffer.alloc(46);
    c.writeUInt32LE(0x02014b50, 0); // signature
    c.writeUInt16LE(20, 4); // version made by
    c.writeUInt16LE(20, 6); // version needed
    c.writeUInt16LE(e.flags, 8);
    c.writeUInt16LE(e.method, 10);
    c.writeUInt16LE(e.time, 12);
    c.writeUInt16LE(e.date, 14);
    c.writeUInt32LE(e.crc, 16);
    c.writeUInt32LE(e.compSize, 20);
    c.writeUInt32LE(e.uncompSize, 24);
    c.writeUInt16LE(e.nameBuf.length, 28);
    c.writeUInt16LE(0, 30); // extra len
    c.writeUInt16LE(0, 32); // comment len
    c.writeUInt16LE(0, 34); // disk number start
    c.writeUInt16LE(0, 36); // internal attrs
    c.writeUInt32LE(0, 38); // external attrs
    c.writeUInt32LE(e.localOffset, 42);
    central.push(c, e.nameBuf);
  }

  const centralSize = central.reduce((n, c) => n + c.length, 0);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0); // signature
  eocd.writeUInt16LE(0, 4); // disk number
  eocd.writeUInt16LE(0, 6); // disk with central dir
  eocd.writeUInt16LE(entries.length, 8); // entries this disk
  eocd.writeUInt16LE(entries.length, 10); // total entries
  eocd.writeUInt32LE(centralSize, 12);
  eocd.writeUInt32LE(centralStart, 16);
  eocd.writeUInt16LE(0, 20); // comment len

  return Buffer.concat([...chunks, ...central, eocd]);
}

// ---- main ----
function main() {
  if (!existsSync(SRC_DIR)) {
    throw new Error(`Source dir not found: ${SRC_DIR}`);
  }
  const files = collectFiles(SRC_DIR);
  if (files.length === 0) {
    throw new Error(`No files found under ${SRC_DIR}`);
  }

  const bytes = buildZip(SRC_DIR, files);
  mkdirSync(dirname(OUT_ZIP), { recursive: true });

  const tmp = `${OUT_ZIP}.tmp`;
  writeFileSync(tmp, bytes);
  if (existsSync(OUT_ZIP)) rmSync(OUT_ZIP);
  renameSync(tmp, OUT_ZIP);

  console.log(`Created ${OUT_ZIP}`);
  console.log(`  entries: ${files.length}`);
  console.log(`  bytes:   ${bytes.length}`);
}

main();
