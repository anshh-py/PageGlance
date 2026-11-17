// Verify the generated Firefox release ZIP: POSIX paths, root entries,
// manifest validity, referenced assets, backend URL presence, and integrity.
import { readFileSync } from 'node:fs';
import { inflateRawSync } from 'node:zlib';

const ZIP = process.env.PACK_OUT_ZIP ||
  `${process.env.USERPROFILE || process.env.HOME}\\Desktop\\PageGlance-Releases\\PageGlance-Firefox-v0.1.0.zip`;

const b = readFileSync(ZIP);

// ---- parse local file entries ----
const entries = [];
let off = 0;
while (off + 4 <= b.length && b.readUInt32LE(off) === 0x04034b50) {
  const nameLen = b.readUInt16LE(off + 26);
  const extraLen = b.readUInt16LE(off + 28);
  const method = b.readUInt16LE(off + 8);
  const compSize = b.readUInt32LE(off + 18);
  const uncompSize = b.readUInt32LE(off + 22);
  const name = b.slice(off + 30, off + 30 + nameLen).toString('utf8');
  const data = b.slice(off + 30 + nameLen + extraLen, off + 30 + nameLen + extraLen + compSize);
  entries.push({ name, method, compSize, uncompSize, data });
  off += 30 + nameLen + extraLen + compSize;
}

const fail = (msg) => { console.error(`FAIL: ${msg}`); process.exitCode = 1; };
const ok = (msg) => console.log(`OK:   ${msg}`);

// 1. path separators
const bs = entries.filter((e) => e.name.includes('\\'));
bs.length === 0 ? ok('no entry contains a backslash') : fail(`backslash entries: ${bs.map((e) => e.name).join(', ')}`);
const nonPosix = entries.filter((e) => !/^[^\\/]+(\/[^\\/]+)*$/.test(e.name));
nonPosix.length === 0 ? ok('all nested paths use forward slashes') : fail(`non-POSIX names: ${nonPosix.map((e) => e.name).join(', ')}`);
ok(`entries: ${entries.length}`);

// 2. root entries
const root = entries.map((e) => e.name).filter((n) => !n.includes('/'));
for (const required of ['manifest.json', 'content.js']) {
  root.includes(required) ? ok(`${required} at archive root`) : fail(`${required} missing from root`);
}
const topLevel = new Set(root.map((n) => n.split('/')[0]));
const unwanted = ['chrome', 'firefox', 'extension'].filter((d) => topLevel.has(d) && !root.includes(d));
unwanted.length === 0 ? ok('no unnecessary parent directory') : fail(`unwanted parent dirs: ${unwanted.join(', ')}`);

// 3. manifest parse + fields
const manifestEntry = entries.find((e) => e.name === 'manifest.json');
if (!manifestEntry) { fail('manifest.json missing'); }

else {
  let m;
  try {
    const raw = manifestEntry.method === 8 ? inflateRawSync(manifestEntry.data) : manifestEntry.data;
    m = JSON.parse(raw.toString('utf8'));
    ok('manifest parses as JSON');
  } catch { fail('manifest is invalid JSON'); }
  if (m) {
    m.manifest_version === 3 ? ok('manifest_version 3') : fail(`manifest_version ${m.manifest_version}`);
    const g = m.browser_specific_settings?.gecko;
    g?.id === 'pageglance-extension@anshh-py.com'
      ? ok(`gecko.id = ${g.id}`) : fail(`gecko.id = ${g?.id}`);
    g?.strict_min_version === '109.0' ? ok(`strict_min_version ${g.strict_min_version}`) : fail(`strict_min_version ${g?.strict_min_version}`);
    const dcp = g?.data_collection_permissions;
    dcp?.required?.includes('websiteContent')
      ? ok(`data_collection_permissions.required = ${JSON.stringify(dcp.required)}`) : fail(`data_collection_permissions = ${JSON.stringify(dcp)}`);
    m.host_permissions?.some((h) => h.includes('ansh9057-pageglance-backend.hf.space'))
      ? ok('host_permissions includes backend') : fail('host_permissions missing backend');
    const cs = m.content_scripts?.[0];
    cs?.js?.includes('content.js') ? ok('content_scripts references content.js') : fail('content_scripts missing content.js');
    m.permissions?.length ? ok(`permissions = ${JSON.stringify(m.permissions)}`) : fail(`permissions = ${JSON.stringify(m.permissions)}`);
  }
}

// 4. referenced assets present
const body = entries.map((e) => e.name).join('\n');
for (const ref of ['index.html', 'chat.html', 'content.js', 'icons/pageglance-16.png', 'icons/pageglance-48.png', 'icons/pageglance-128.png']) {
  body.includes(ref) ? ok(`referenced asset present: ${ref}`) : fail(`missing referenced asset: ${ref}`);
}

// 5. backend URL / no localhost across all entries
const allText = entries.map((e) => {
  try { return e.method === 8 ? inflateRawSync(e.data).toString('utf8') : e.data.toString('utf8'); } catch { return ''; }
}).join('\n');
allText.includes('https://ansh9057-pageglance-backend.hf.space')
  ? ok('production backend URL present in archive') : fail('backend URL not found in archive');
/127\.0\.0\.1|localhost|DEV_SENTINEL/.test(allText)
  ? fail('localhost / dev fallback found in archive') : ok('no localhost/dev fallback in archive');

// 6. integrity: re-inflate every deflated entry
let integrityOk = true;
for (const e of entries) {
  if (e.method === 8) {
    try {
      const out = inflateRawSync(e.data);
      if (out.length !== e.uncompSize) { integrityOk = false; fail(`inflate size mismatch for ${e.name}`); }
    } catch { integrityOk = false; fail(`inflate failed for ${e.name}`); }
  }
}
if (integrityOk) ok('all deflated entries inflate cleanly (integrity OK)');

console.log(`\nZIP: ${ZIP}`);
console.log(`size: ${b.length} bytes`);
