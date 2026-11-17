import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readFileSync } from 'node:fs'

// --- Browser abstraction tests ---

test('browser abstraction exports browser object with storage, runtime, tabs, scripting', async () => {
  // Reset module-level state by re-importing is not possible in node:test,
  // but we can verify the shape of the exported object.
  const { browser } = await import('./browser.ts')
  assert.ok(browser)
  assert.ok(browser.storage)
  assert.ok(browser.storage.local)
  assert.ok(typeof browser.storage.local.get === 'function')
  assert.ok(typeof browser.storage.local.set === 'function')
  assert.ok(typeof browser.storage.local.remove === 'function')
  assert.ok(browser.runtime)
  assert.ok(typeof browser.runtime.getURL === 'function')
  assert.ok(browser.runtime.onMessage)
  assert.ok(typeof browser.runtime.onMessage.addListener === 'function')
  assert.ok(browser.tabs)
  assert.ok(typeof browser.tabs.query === 'function')
  assert.ok(typeof browser.tabs.create === 'function')
  assert.ok(typeof browser.tabs.get === 'function')
  assert.ok(typeof browser.tabs.sendMessage === 'function')
  assert.ok(browser.scripting)
  assert.ok(typeof browser.scripting.executeScript === 'function')
})

test('browser abstraction exports isFirefox boolean', async () => {
  const { isFirefox } = await import('./browser.ts')
  assert.ok(typeof isFirefox === 'boolean')
})

// --- Manifest validation tests ---

function loadManifest(browserName) {
  const path = `manifests/${browserName}.json`
  const content = readFileSync(path, 'utf8')
  return JSON.parse(content)
}

test('chrome manifest is valid Manifest V3 with required fields', () => {
  const manifest = loadManifest('chrome')
  assert.equal(manifest.manifest_version, 3)
  assert.equal(manifest.name, 'PageGlance')
  assert.equal(manifest.version, '0.1.0')
  assert.ok(manifest.action)
  assert.equal(manifest.action.default_popup, 'index.html')
  assert.ok(manifest.permissions)
  assert.ok(manifest.permissions.includes('storage'))
  assert.ok(manifest.permissions.includes('scripting'))
  assert.ok(manifest.permissions.includes('activeTab'))
  assert.ok(manifest.host_permissions)
  assert.ok(manifest.host_permissions.some((p) => p.includes('localhost')))
  assert.ok(manifest.host_permissions.some((p) => p.includes('127.0.0.1')))
  assert.ok(
    manifest.host_permissions.some((p) => p.includes('ansh9057-pageglance-backend.hf.space')),
  )
  // No broad wildcard permissions
  assert.ok(!manifest.host_permissions.includes('https://*'))
  assert.ok(!manifest.host_permissions.includes('<all_urls>'))
  assert.ok(manifest.icons)
  assert.ok(manifest.icons['16'])
  assert.ok(manifest.icons['48'])
  assert.ok(manifest.icons['128'])
})

test('firefox manifest is valid Manifest V3 with gecko settings', () => {
  const manifest = loadManifest('firefox')
  assert.equal(manifest.manifest_version, 3)
  assert.equal(manifest.name, 'PageGlance')
  assert.equal(manifest.version, '0.1.1')
  assert.ok(manifest.action)
  assert.equal(manifest.action.default_popup, 'index.html')
  assert.ok(manifest.permissions)
  assert.ok(manifest.permissions.includes('storage'))
  assert.ok(manifest.permissions.includes('scripting'))
  assert.ok(manifest.permissions.includes('activeTab'))
  assert.ok(manifest.host_permissions)
  assert.ok(
    manifest.host_permissions.some((p) => p.includes('ansh9057-pageglance-backend.hf.space')),
  )
  // Production Firefox manifest has no localhost/127.0.0.1 host permissions
  assert.ok(!manifest.host_permissions.some((p) => p.includes('localhost')))
  assert.ok(!manifest.host_permissions.some((p) => p.includes('127.0.0.1')))
  // No broad wildcard permissions
  assert.ok(!manifest.host_permissions.includes('https://*'))
  assert.ok(!manifest.host_permissions.includes('<all_urls>'))
  // Firefox-specific gecko settings
  assert.ok(manifest.browser_specific_settings)
  assert.ok(manifest.browser_specific_settings.gecko)
  assert.ok(manifest.browser_specific_settings.gecko.id)
  assert.ok(manifest.browser_specific_settings.gecko.strict_min_version)
})

test('chrome and firefox manifests have consistent core fields', () => {
  const chrome = loadManifest('chrome')
  const firefox = loadManifest('firefox')
  assert.equal(chrome.name, firefox.name)
  // Firefox and Chrome versions are intentionally independent (Firefox 0.1.1 vs Chrome 0.1.0)
  assert.equal(chrome.manifest_version, firefox.manifest_version)
  assert.deepEqual(chrome.permissions, firefox.permissions)
  assert.deepEqual(chrome.action.default_popup, firefox.action.default_popup)
  assert.deepEqual(chrome.icons, firefox.icons)
})
